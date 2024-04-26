from fastapi import FastAPI, HTTPException
import subprocess
import os
from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7")
pipe.to(torch_device="cuda:1", torch_dtype=torch.float32)

app = FastAPI()
pid = os.getpid()


@app.post("/")
async def root(request: dict):
    prompt = request['prompt']
    pipe(
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
                    "Server": "8083",
                    "GPUload": round(gpu_utilization_decimal, 4)
                }
        return {"Server": "8083", "GPUload": 0.00}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get GPU load: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8083)
