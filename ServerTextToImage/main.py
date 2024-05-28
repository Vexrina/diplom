import asyncio
from fastapi import FastAPI
from diffusers import DiffusionPipeline
import torch
import uvicorn

PORT = 8083

pipe = DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7")
pipe.to(torch_device="cuda:1", torch_dtype=torch.float16)

app = FastAPI()

NUM_WORKERS = 2
ACTIVE_WORKERS = 0


async def create_image(prompt: str):
    await asyncio.to_thread(
        pipe,
        prompt=prompt,
        num_inference_steps=1,
        guidance_scale=8.0,
        lcm_origin_steps=25,
        output_type="pil",
    )


@app.post("/")
async def root(request: dict):
    global ACTIVE_WORKERS
    ACTIVE_WORKERS += 1
    prompt = request['prompt']
    task = create_image(prompt)
    try:
        await asyncio.wait_for(task, timeout=60*3)
    except asyncio.TimeoutError:
        print("gave up wait, cancel task")
        ACTIVE_WORKERS -= 1
        return {"output": "cancelling"}
    ACTIVE_WORKERS -= 1
    return {"output": "success"}


@app.get("/load")
async def load():
    global ACTIVE_WORKERS, NUM_WORKERS
    utilization = round(ACTIVE_WORKERS/NUM_WORKERS, 4)

    return {
        "Server": PORT,
        "GPUload": utilization
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=PORT)
