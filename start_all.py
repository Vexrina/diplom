import subprocess

commands = [
    "go run ./LBServer/main.go",
    "python ./LBServer/lbmodel.py",
    "go run ./ServerTextToBoth/main.go",
    "go run ./ServerTextToImage/main.go",
    "go run ./ServerTextToVideo/main.go",
]

processes = []

for command in commands:
    process = subprocess.Popen(command, shell=True)
    print(f"{command} have started")
    processes.append(process)

print("All processes have started")

for process in processes:
    process.wait()


print("All processes have finished")
