from fastapi import FastAPI, HTTPException
import subprocess
import os
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import torch

VideoPipe = DiffusionPipeline.from_pretrained(
    "damo-vilab/text-to-video-ms-1.7b",
    torch_dtype=torch.float16,
    variant="fp16",
)
VideoPipe.to(torch_device="cuda:1", torch_dtype=torch.float32)
VideoPipe.scheduler = DPMSolverMultistepScheduler.from_config(
    VideoPipe.scheduler.config
)
VideoPipe.enable_model_cpu_offload()

app = FastAPI()
pid = os.getpid()


@app.post("/")
async def root(request: dict):
    prompt = request['prompt']
    VideoPipe(
        prompt=prompt,
        num_inference_steps=25,
    )
    return {"output": "success"}


@app.get("/load")
async def load():
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits",
            ],
            shell=True,
        )
        lines = out.decode('utf-8').split('\n')
        for line in lines:
            fields = line.strip().split(",")
            if len(fields) >= 2 and fields[0] == str(pid):
                gpu_utilization = float(fields[1])
                gpu_utilization_decimal = gpu_utilization / 40960.0
                return {
                    "Server": "8084",
                    "GPUload": round(gpu_utilization_decimal, 4)
                }
        return {"Server": "8084", "GPUload": 0.00}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get GPU load: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8084)
