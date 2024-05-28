def min_min_realtime_scheduling(server_loads, request) -> int:
    video = request['video']

    if video:
        possible_servers = [0, 1]
    else:
        possible_servers = [1, 2]

    # Найти сервер с минимальной нагрузкой из возможных серверов
    min_load = float('inf')
    selected_server = -1

    for server in possible_servers:
        if server_loads[server] < min_load:
            min_load = server_loads[server]
            selected_server = server

    server_loads[selected_server] += 1

    return selected_server
