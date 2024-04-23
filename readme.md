# Выпускная квалификационная работа бакалавра
# Bachelor's final qualifying work

## Тема: Разработка и анализ алгоритмов обучения с подкреплением для оптимизации динамического выделения ресурсов GPU в серверных системах
## Topic: Development and analysis of reinforcement learning algorithms to optimize the dynamic allocation of GPU resources in server systems

### Выполнил: Щавлев Константин Владимирович, 2024
### Performed by: Konstantin Vladimirovich Shchavlev, 2024


Дорожная Карта / RoadMap
---
- [X] Реализовать доступ из GoLang к диффузионным моделям на Python, не рождая процессы / Implement access from GoLang to diffusion models in Python without creating processes
- [X] Реализовать серверы с таким доступом / Implement servers with such access
- [X] Реализовать LB сервер, распределяющий запросы между серверами моделей / Implement an LB server that distributes requests between model servers
- [X] Реализовать модель обучения с подкреплением для классификации запросов по серверам с оптимизацией нагрузки / Implement a RL model for classifying requests across servers with load optimization
- [X] Реализовать общение между LB сервером и RL моделью посредством ~~Kafka~~ RestAPI / Implement communication between the LB server and the RL model via ~~Kafka~~ RestApi
- [ ] Реализовать симулятор запросов, которые будут посылаться на LB сервер для распределения / Implement a simulator of requests that will be sent to the LB server for distribution
- [ ] Обучить модель используя симулятор / Train a model using a simulator
- [ ] Провести стресс-тест модели / Conduct a stress test of the model
- [ ] Сравнить результаты стресс-теста модели и обычного алгоритма LB / Compare the results of the stress test of the model and the usual LB algorithm