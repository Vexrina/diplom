import asyncio
from fastapi import FastAPI, HTTPException
import httpx
import numpy as np
import time
import torch
import torch.nn as nn
from torch import optim
import uvicorn
import lbmodel
import minmin


miniminFlag = False

time_threshold = 60*3
time_bonus = 1.3

server_efficiency_bonus = 1.25

fine = -5
basic_reward = 2.5
save_string = f"server_eff_{server_efficiency_bonus}_time_bon_{time_bonus}_reward_{basic_reward}_fine_{fine}"

app = FastAPI()

#  block with constants
IN_SIZE = 6  # 7 параметров входных данных
OUT_SIZE = 3  # 3 параметра выходных данных
LR = 1e-3
num_episodes = 1000
GAMMA = torch.tensor(0.9)
# For dont use very much GPUspace
MAX_PER_SERVER = 3
requests_per_server = {0: 0, 1: 0, 2: 0}
MAX_WORKERS = 4

# create device
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = "cpu"

# create a model, criterion, optimizer, callbacks
model = lbmodel.RLModel(
    input_size=IN_SIZE,
    output_size=OUT_SIZE,
).to(device)

# checkpoint = torch.load(f'{save_string}_episodes_15.pt')
# model.load_state_dict(checkpoint['model_state_dict'])

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

callback = lbmodel.CustomCallback(model, optimizer, save_string)

callbacks = [callback]


max_len_queue = 50
request_queue = asyncio.Queue()
request_events = {}
responses = {}

response_time = np.array([])


def get_environment_state(request: dict | None) -> dict:
    """Функция-забиратель нагрузки на серверы.
Ходит по внешним серверам(в рамках ВКР рассматриваются исключительно
localhost:808(1,2,3), но в теории можно ее масштабировать достаточно просто).

Заполняет нагрузку на сервера в слайс, высчитывает среднее время отклика и
считает текущую заполненность очереди.

    Args:
        request (dict): Запрос пользователя, словарь в формате
        {"prompt": str, "video":bool}

    Returns:
        dict в формате:

        ServerLoads        []float64

        RequestQueueLength int

        ResponseTime       float64

        prompt             str

        video              bool
    """
    global request_queue
    global request_events
    global response_time
    server_urls = [
        "http://localhost:8084/load",
        "http://localhost:8082/load",
        "http://localhost:8083/load",
    ]
    server_loads = []

    for url in server_urls:
        try:
            response = httpx.get(url)
            server_loads.append(float(response.json()['GPUload']))
        except Exception as e:
            print(f"Error getting server load from {url}: {e}")
            server_loads.append(1.0)

    mrt = np.mean(response_time) if response_time.size > 0 else 0
    request_queue_length = request_queue.qsize() / max_len_queue
    if request:
        return {
            "server_loads": server_loads,
            "queue_len": request_queue_length,
            "mrt": mrt,
            "prompt": request["prompt"],
            "video": request["video"],
        }
    return {
        "server_loads": server_loads,
        "queue_len": request_queue_length,
        "mrt": mrt,
    }


def send_reward(
    request: dict,
    server_to_redirect: int,
    timer: float,
    max_workers_per_server: bool = False,
):
    """Отправляет награду в модельку

    Args:
        request (dict): изначальный запрос пользователя
        server_to_redirect (int): сервер, на который перенаправили запрос
        timer (float): время ответа от сервера.
    """
    environment = get_environment_state(request)
    reward = calculate_reward(
        request,
        server_to_redirect,
        environment['server_loads'],
        timer
    )
    if max_workers_per_server:
        reward -= fine
    lbmodel.update_model(
        reward={
            "reward": reward,
            "environment": environment,
        },
        model=model,
        gamma=GAMMA,
        criterion=criterion,
        callbacks=callbacks,
        optimizer=optimizer,
        device=device,
    )


def calculate_reward(
    req: dict,
    server_to_redirect: int,
    server_loads: list[float],
    timer: float,
) -> float:
    """Подсчитывает награду для модели.

    Args:
        req (dict): запрос пользователя
        server_to_redirect (int): на какой сервер послали запрос
        server_loads (list[float]): нагрузка на сервера
        timer (float): время ответа от сервера

    Returns:
        float: награда
    """
    reward = 0.0

    if req['video']:
        if server_to_redirect == 2:
            reward -= fine
        else:
            reward += basic_reward
    else:
        if server_to_redirect == 0:
            reward -= fine
        else:
            reward += basic_reward

    if timer < time_threshold:
        reward += time_bonus

    reward += max(1-np.mean(server_loads), 0) * server_efficiency_bonus

    return reward


@app.post("/")
async def main_job(request_body: dict):
    """
Обработчик запросов.

Является основным эндпоинтом проекта, т.к. через него происходит общение
всего со всем, т.е. в контексте облачных вычислений является сервером
посредником между клиентом и системой облачных комьютеров.

Разбирает очередь - worker.

    Args:
        request_body (dict): При появлении запроса расшифровывает его в
        RequestBody, после чего добавляет его в очередь запросов.

    Raises:
        HTTPException: Если очередь переполнена ->
        http.StatusServiceUnavailable

    Returns:
        dict: Как только получен ответ, возвращает его пользователю.
    """
    global request_queue
    global request_events
    if request_queue.qsize() >= max_len_queue:
        raise HTTPException(
            status_code=500,
            detail="Request queue is full",
        )

    event = asyncio.Event()
    request_events[id(request_body)] = event
    await request_queue.put(request_body)

    await event.wait()
    env_state = get_environment_state(None)
    del request_events[id(request_body)]

    response_data = responses.get(request_body['prompt'])
    if response_data is None:
        raise HTTPException(
            status_code=404,
            detail="Response data not found for the given prompt"
        )

    response = {
        'result': {
            'status_code': response_data.status_code,
            'content': response_data.content.decode(),
        },
        "server_loads": env_state["server_loads"],
        "queue_len": env_state["queue_len"],
        "mrt": env_state["mrt"],
    }
    return response


async def choose_server(env, req):
    if miniminFlag:
        server_redirect = minmin.min_min_realtime_scheduling(
            env["server_loads"],
            req
        )
    else:
        server_redirect = lbmodel.choose_server(env, model, device)

    match server_redirect:
        case 0:
            return "8084", server_redirect
        case 1:
            return "8082", server_redirect
        case 2:
            return "8082", server_redirect
        case _:
            return ""


async def worker():
    """
Используем воркера, который при появлении запроса будет запускать логику
его обработки. Он будет засекать время начала обработки, ходить в нейронку
со средой и запросом, принимать выход нейронки. После чего отсылать запрос
на нужный сервер и отправлять награду и новую среду нейронке для обновления
весов.

Когда запрос будет обработан вернет его клиенту.

В Golang запускается как горутина
    """
    global response_time
    server_url = "http://localhost:"
    print("Worker started")
    while True:
        req = await request_queue.get()
        event = request_events.get(id(req))
        start = time.time()
        env = get_environment_state(req)
        print(env)
        port, server_redirect = choose_server(env, req)

        MaxRequest = requests_per_server[server_redirect] >= MAX_PER_SERVER

        if MaxRequest:
            err = f"Maximum requests reached for server {server_redirect}"
            resp = httpx.Response({
                "error": err,
            })
            responses[req['prompt']] = resp
            if event:
                event.set()
            end = time.time()-start
            response_time = np.append(response_time, end)
            print("send reward")
            if not miniminFlag:
                send_reward(req, server_redirect, end+1, True)
        else:
            try:
                requests_per_server[server_redirect] += 1
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        url=server_url + port,
                        json=req,
                        timeout=time_threshold
                    )
                    resp = await client.get('https://google.com')
                responses[req['prompt']] = resp
                if event:
                    event.set()
                end = time.time()-start
                response_time = np.append(response_time, end)
                requests_per_server[server_redirect] -= 1
                if not miniminFlag:
                    send_reward(req, server_redirect, end)
            except httpx.ReadTimeout as e:
                responses[req['prompt']] = httpx.Response(
                    status_code=500,
                    content=str(e)
                )
                if event:
                    event.set()
                end = time.time()-start
                response_time = np.append(response_time, end)
                requests_per_server[server_redirect] -= 1
                if not miniminFlag:
                    send_reward(req, server_redirect, end+1)
        request_queue.task_done()
        await asyncio.sleep(0.1)


async def main():
    config = uvicorn.Config(app, host="0.0.0.0", port=8090)
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    tasks = [loop.create_task(main())]
    for i in range(MAX_WORKERS):
        tasks.append(loop.create_task(worker()))
    loop.run_until_complete(asyncio.gather(*tasks))
