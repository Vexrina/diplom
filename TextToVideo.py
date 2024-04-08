import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")

pipe.to(torch_device="cuda", torch_dtype=torch.float32)

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

pipe.enable_model_cpu_offload()

prompt = "Spiderman is surfing"
video_frames = pipe(
    prompt=prompt,
    num_inference_steps=25
    ).frames[0]

print("Success")
