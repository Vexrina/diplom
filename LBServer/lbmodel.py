import torch
import torch.nn as nn
import numpy as np
import csv


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
        self.episodes = 0

    def on_episode_end(self):
        torch.save(self.model.state_dict(
        ), f'server_eff_125_time_bon_13_reward_25_fine_5_episodes_{self.episodes}.pt')
        with open(f'server_eff_125_time_bon_13_reward_25_fine_5_episodes_{self.episodes//527}.csv', mode='w') as csv_file:
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
        self.rewards.append(reward.item())


def process_data(data: dict) -> dict:
    """Предобработка данных, полученных от Golang серверов.
    Данные приходят в "грязном" формате, т.е. имеются вложенные словари.
    В данной функции они "очищаются", т.е. мы их разкручиваем до тех пор, пока
    как бы не примут вид одной строки.

    Args:
        data (dict): набор всех данных в "грязном" формате.

    Returns:
        processed_data (list): чистый набор данных
    """
    processed_data = {
        'server1_loads': data['server_loads'][0],
        'server2_loads': data['server_loads'][1],
        'server3_loads': data['server_loads'][2],
        'request_queue_length': data['queue_len'],
        'response_time': data['mrt'],
        # 'prompt': data['request_body']['prompt'],
        'video': 1 if data['video'] else 0,
    }
    return list(processed_data.values())


def select_action(state, model, device):
    state_tensor = torch.tensor(state, dtype=torch.float32)
    with torch.no_grad():
        action_values = model(state_tensor.to(device))
    action = np.argmax(action_values.cpu().numpy())
    return action


def choose_server(server_data: dict, model: RLModel, device: str):
    processed_data = process_data(server_data)
    action = select_action(
        processed_data,
        model,
        device,
    )
    return action


def update_model(
    reward: dict,
    model: RLModel,
    gamma: torch.Tensor,
    criterion,    # MSELoss
    callbacks,    # [CustomCallback, etc.]
    optimizer,    # nn.Optimizer
    device: str,  # cuda or cpu
):
    data = process_data(reward['environment'])
    reward_value = torch.tensor(
        reward['reward'],
        dtype=torch.float32,
    ).to(device)
    with torch.no_grad():
        target = reward_value + gamma.to(device) * torch.max(
            model(
                torch.tensor(
                    data,
                    dtype=torch.float32
                ).to(device)
            )
        )
    current_prediction = model(
        torch.tensor(
            data, dtype=torch.float32
        ).to(device)
    )
    current_prediction_for_action = current_prediction[latest_action]

    # Считаем ошибку
    loss = criterion(current_prediction_for_action, target)

    # отдаем это в коллбэки
    for callback in callbacks:
        callback.episodes += 1
        callback.on_loss_calculated(loss)
        callback.on_reward_received(reward_value)
        if callback.episodes % 527 == 0:
            callback.on_episode_end()

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
