import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI()
latest_action = None
episode = 0


class RLModel(nn.Module):
    """
    RL модель использущаяся в дипломе.
    """
    def __init__(
        self,
        input_size: int,
        output_size: int,
    ):
        """

        Args:
            input_size (int): Размерность данных, приходящая на обработку
            output_size (int): Количество серверов, среди которых выбираем.
        """
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
        )

    def forward(self, x):
        return self.body(x)


class CustomCallback:
    def __init__(self, model):
        self.model = model
        self.losses = []
        self.rewards = []
        self.episode = 0

    def on_episode_end(self):
        torch.save(self.model.state_dict(), f'model_weights_{episode}.pt')
        with open('model_data.csv', mode='w') as csv_file:
            self.episode = 0
            fieldnames = ['episode', 'loss', 'reward']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for loss, reward in zip(self.losses, self.rewards):
                self.episode += 1
                writer.writerow(
                    {'episode': episode, 'loss': loss, 'reward': reward}
                )

    def on_loss_calculated(self, loss):
        self.losses.append(loss.item())

    def on_reward_received(self, reward):
        self.rewards.append(reward)


#  block with constants
input_size = 7  # 7 параметров входных данных
output_size = 3  # 3 параметра выходных данных
learning_rate = 1e-3
num_episodes = 1000
gamma = 0.9

# create device
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# create a model, criterion, optimizer, callbacks
model = RLModel(
    input_size=input_size,
    output_size=output_size,
).to_device(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
callback = CustomCallback(model)

callbacks = [callback]


class RequestBody(BaseModel):
    prompt: str
    video: bool


class Data(BaseModel):
    server_loads: list[float]
    request_queue_length: float
    response_time: float
    request_body: RequestBody


class RewardRequest(BaseModel):
    reward: float
    environment: Data


def process_data(data: Data) -> dict:
    """Предобработка данных, полученных от Golang серверов.
    Данные приходят в "грязном" формате, т.е. имеются вложенные словари.
    В данной функции они "очищаются", т.е. мы их разкручиваем до тех пор, пока
    как бы не примут вид одной строки.

    Args:
        data (dict): набор всех данных в "грязном" формате.

    Returns:
        processed_data (dict): чистый набор данных
    """
    processed_data = {
        'server1_loads': data['server_loads'][0],
        'server2_loads': data['server_loads'][1],
        'server3_loads': data['server_loads'][2],
        'request_queue_length': data['request_queue_length'],
        'response_time': data['response_time'],
        'prompt': data['request_body']['prompt'],
        'video': data['request_body']['video'],
    }
    return processed_data


def select_action(state, model):
    state_tensor = torch.tensor(state, dtype=torch.float32)
    with torch.no_grad():
        action_values = model(state_tensor)
    action = np.argmax(action_values.numpy())
    return action


def choose_server(server_data: Data):
    processed_data = process_data(server_data)
    action = select_action(
            processed_data,
            model,
        )
    return action


@app.post("/predict_server")
def predict_server(server_data: Data):
    try:
        server_id = choose_server(server_data=server_data)
        return {"server": f"server{server_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def update_model(reward: RewardRequest):
    data = process_data(RewardRequest.environment)
    reward_value = reward.reward
    with torch.no_grad():
        target = reward_value + gamma * torch.max(
            model(
                torch.tensor(
                    data,
                    dtype=torch.float32
                )
            )
        )
    current_prediction = model(torch.tensor(data, dtype=torch.float32))
    current_prediction_for_action = current_prediction[latest_action]

    # Считаем ошибку
    loss = criterion(current_prediction_for_action, target)

    # отдаем это в коллбэки
    for callback in callbacks:
        callback.on_loss_calculated(loss)
        callback.on_reward_received(reward)
        callback.on_episode_end()

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


@app.post("/get_reward")
def get_reward(reward: RewardRequest):
    try:
        update_model(reward)
        return {"sucess": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=(str(e)))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7999)
