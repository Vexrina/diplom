import asyncio
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from fastapi import FastAPI
import torch
import uvicorn


PORT = 8082

VideoPipe = DiffusionPipeline.from_pretrained(
    "damo-vilab/text-to-video-ms-1.7b",
    torch_dtype=torch.float16,
    variant="fp16",
)
VideoPipe.to(torch_device="cuda:1", torch_dtype=torch.float16)
VideoPipe.scheduler = DPMSolverMultistepScheduler.from_config(
    VideoPipe.scheduler.config
)
VideoPipe.enable_model_cpu_offload()

ImagePipe = DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7")
ImagePipe.to(torch_device="cuda:1", torch_dtype=torch.float16,)

app = FastAPI()

NUM_WORKERS = 4
ACTIVE_WORKERS = 0
LOCK = asyncio.Lock()


async def create_image(prompt: str):
    await asyncio.to_thread(
        ImagePipe,
        prompt=prompt,
        num_inference_steps=1,
        guidance_scale=8.0,
        lcm_origin_steps=25,
        output_type="pil",
    )


async def create_video(prompt: str):
    await asyncio.to_thread(
        VideoPipe,
        prompt=prompt,
        num_inference_steps=25,
    )


@app.post("/")
async def root(request: dict):
    global ACTIVE_WORKERS
    ACTIVE_WORKERS += 1
    prompt, video = request['prompt'], request['video']

    if video:
        task = create_video(prompt)
    else:
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
