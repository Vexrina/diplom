from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7")

pipe.to(torch_device="cuda:1", torch_dtype=torch.float32)

prompt = "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"

num_inference_steps = 4

images = pipe(
    prompt=prompt,
    num_inference_steps=num_inference_steps,
    guidance_scale=8.0,
    lcm_origin_steps=50,
    output_type="pil"
    ).images
print("Success")
