import torch
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel, BitsAndBytesConfig
import imageio as iio
import numpy as np
from PIL import Image  # Import PIL for image processing
import io
import time

torch.manual_seed(42)

prompt_template = {
    "template": (
        "<|start_header_id|>system<|end_header_id|>\n\nDescribe the video by detailing the following aspects: "
        "1. The main content and theme of the video."
        "2. The color, shape, size, texture, quantity, text, and spatial relationships of the contents, including objects, people, and anything else."
        "3. Actions, events, behaviors temporal relationships, physical movement changes of the contents."
        "4. Background environment, light, style, atmosphere, and qualities."
        "5. Camera angles, movements, and transitions used in the video."
        "6. Thematic and aesthetic concepts associated with the scene, i.e. realistic, futuristic, fairy tale, etc<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
    ),
    "crop_start": 95,
}

def export_to_video_bytes(fps, frames):
    request = iio.core.Request("<bytes>", mode="w", extension=".mp4")
    pyavobject = iio.get_writer(request, format='mp4', codec="libx264", fps=fps)
    
    # Convert each PIL Image to a NumPy array and write it to the video
    for frame in frames:
        if not isinstance(frame, np.ndarray):
            frame = np.array(frame)  # Convert PIL Image to NumPy array
        pyavobject.append_data(frame)
    
    pyavobject.close()
    return request.get_result()

def export_to_video(frames, path, fps):
    video_bytes = export_to_video_bytes(fps, frames)
    with open(path, "wb") as f:
        f.write(video_bytes)

model_id = "tencent/HunyuanVideo"
quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4", llm_int8_skip_modules=["proj_out", "norm_out"])
transformer = HunyuanVideoTransformer3DModel.from_pretrained(
    model_id, subfolder="transformer", torch_dtype=torch.bfloat16, revision="refs/pr/18", quantization_config=quantization_config
)
pipe = HunyuanVideoPipeline.from_pretrained(model_id, transformer=transformer, torch_dtype=torch.float16, revision="refs/pr/18")
pipe.scheduler._shift = 7.0
pipe.vae.enable_tiling()
pipe.enable_model_cpu_offload()

start_time = time.perf_counter()
output = pipe(
    prompt="a cat walks along the sidewalk of a city. The camera follows the cat at knee level. The city has many people and cars moving around, with advertisement billboards in the background",
    height=720,
    width=1280,
    num_frames=45,
    prompt_template=prompt_template,
    num_inference_steps=15,
).frames
export_to_video(output, "output.mp4", fps=15)
print("Time:", round(time.perf_counter() - start_time, 2), "seconds")
print("Max vram:", round(torch.cuda.max_memory_allocated(device="cuda") / 1024 ** 3, 3), "GiB")