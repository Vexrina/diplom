import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException
import httpx
import numpy as np
import time
import torch
import torch.nn as nn
from torch import optim
import uvicorn
import lbmodel


app = FastAPI()

#  block with constants
IN_SIZE = 6  # 7 параметров входных данных
OUT_SIZE = 3  # 3 параметра выходных данных
LR = 1e-3
num_episodes = 1000
GAMMA = torch.tensor(0.9)
# For dont use very much GPUspace
MAX_REQUESTS_PER_SERVER = 5
requests_per_server = {0: 0, 1: 0, 2: 0}

# create device
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# create a model, criterion, optimizer, callbacks
model = lbmodel.RLModel(
    input_size=IN_SIZE,
    output_size=OUT_SIZE,
).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
callback = lbmodel.CustomCallback(model)

callbacks = [callback]

max_len_queue = 50
request_queue = deque()
request_events = {}
responses = {}

response_time = np.array([])

time_threshold = 4 * 60
time_bonus = 1.75

server_efficiency_bonus = 1.25


def get_environment_state(request: dict) -> dict:
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

    mrt = np.mean(response_time) if response_time.size > 0 else 0
    request_queue_length = len(request_queue) / max_len_queue

    return {
        "server_loads": server_loads,
        "queue_len": request_queue_length,
        "mrt": mrt,
        "prompt": request["prompt"],
        "video": request["video"],
    }


def send_reward(request: dict, server_to_redirect: int, timer: float):
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
        timer,
    )
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

    # не на тот сервер
    if req['video']:
        if server_to_redirect == 2:
            reward -= 3
        else:
            reward += 0.5
    else:
        if server_to_redirect == 0:
            reward -= 3
        else:
            reward += 0.5

    # ответ от сервера уже не актуален
    if timer < time_threshold:
        reward += time_bonus

    # эффективное использование серверов
    reward += max(1-np.mean(server_loads), 0)*server_efficiency_bonus

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
    if len(request_queue) >= max_len_queue:
        raise HTTPException(
            status_code=500,
            detail="Request queue is full",
        )

    event = asyncio.Event()
    request_events[id(request_body)] = event
    request_queue.append(request_body)

    await event.wait()
    del request_events[id(request_body)]
    
    response = responses[request_body['prompt']]
    return response.json()


async def worker():
    """
Используем воркера, который при появлении запроса будет запускать логику
его обработки. Он будет засекать время начала обработки, ходить в нейронку
со средой и запросом, принимать выход нейронки. После чего отсылать запрос
на нужный сервер и отправлять награду и новую среду нейронке для обновления
весов.

Когда запрос будет обработан вернет его клиенту.

В Golang запускается как корутина
    """
    server_url = "http://localhost:"
    print("Worker started")
    while True:
        if request_queue:
            print("get request")
            req = request_queue.popleft()
            event = request_events.get(id(req))
            start = time.time()
            env = get_environment_state(req)
            server_redirect = lbmodel.choose_server(env, model, device)
            print("get redirect server")
            match server_redirect:
                case 0:
                    port = "8084"
                case 1:
                    port = "8082"
                case 2:
                    port = "8082"
                case _:
                    continue

            if requests_per_server[server_redirect] >= MAX_REQUESTS_PER_SERVER:
                err = f"Maximum requests reached for server {server_redirect}"
                resp = {
                    "error": err,
                }
                responses[req['prompt']] = resp
                if event:
                    event.set()
                end = time.time()-start
                np.append(response_time, end)
                reward = calculate_reward(
                    req, server_redirect
                )
                send_reward(reward-2, req)
            else:
                requests_per_server[server_redirect] += 1
                print("wait answer from server:", port)
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        url=server_url + port,
                        json=req,
                        timeout=3*60
                    )
                responses[req['prompt']] = resp
                if event:
                    event.set()
                end = time.time()-start
                np.append(response_time, end)
                print("send reward")
                requests_per_server[server_redirect] -= 1
                send_reward(req, server_redirect, end)
        else:
            print("dont get request")
            await asyncio.sleep(0.5)


async def main():
    # Запуск сервера uvicorn без использования uvicorn.run()
    config = uvicorn.Config(app, host="0.0.0.0", port=8080)
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=2)  # Запускаем в двух потоках
    tasks = [
        loop.create_task(main()),
        loop.create_task(worker()),
    ]
    loop.run_until_complete(asyncio.gather(*tasks))
