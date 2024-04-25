from fastapi import FastAPI, HTTPException
import subprocess
import os
from typing import Optional
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

ImagePipe = DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7")
ImagePipe.to(torch_device="cuda:1", torch_dtype=torch.float32,)

app = FastAPI()
pid = os.getpid()


@app.post("/")
async def root(prompt: str, video: Optional[bool] = False):
    if video:
        VideoPipe(
            prompt=prompt,
            num_inference_steps=25,
        )
    else:
        ImagePipe(
            prompt=prompt,
            num_inference_steps=4,
            guidance_scale=8.0,
            lcm_origin_steps=50,
            output_type="pil",
        )
    return {"output": "success"}


@app.get("/load")
async def load():
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            universal_newlines=True,
        )
        lines = out.split("\n")
        for line in lines:
            fields = line.strip().split(",")
            if len(fields) >= 2 and fields[0] == str(pid):
                gpu_utilization = float(fields[1])
                gpu_utilization_decimal = gpu_utilization / 100.0
                return {
                    "Server": "8082",
                    "GPUload": round(gpu_utilization_decimal, 4)
                    }
        return {"Server": "8082", "GPUload": 0.00}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get GPU load: {str(e)}"
            )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8082)
