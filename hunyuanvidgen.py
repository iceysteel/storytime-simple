#!/usr/bin/env python
# coding: utf-8

import torch
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel, BitsAndBytesConfig
import imageio as iio
import math
import numpy as np
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

model_id = "tencent/HunyuanVideo"
quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4", llm_int8_skip_modules=["proj_out", "norm_out"])
transformer = HunyuanVideoTransformer3DModel.from_pretrained(
    model_id, subfolder="transformer", torch_dtype=torch.bfloat16, revision="refs/pr/18", quantization_config=quantization_config
)
pipe = HunyuanVideoPipeline.from_pretrained(model_id, transformer=transformer, torch_dtype=torch.float16, revision="refs/pr/18")
pipe.scheduler._shift = 7.0
pipe.vae.enable_tiling()
pipe.enable_model_cpu_offload()

output = pipe(
    prompt="a cat walks along the sidewalk of a city. The camera follows the cat at knee level. The city has many people and cars moving around, with advertisement billboards in the background",
    height=720,
    width=480,
    num_frames=120,
    prompt_template=prompt_template,
    num_inference_steps=15,
).frames[0]

def export_to_video_bytes(fps, frames):
    with io.BytesIO() as buffer:
        writer = iio.get_writer(buffer, format='mp4', mode='I', fps=fps)
        for frame in frames:
            np_frame = np.array(frame)
            writer.append_data(np_frame)
        writer.close()
        video_bytes = buffer.getvalue()
    return video_bytes
def export_to_video(frames, path, fps):
    video_bytes = export_to_video_bytes(fps, frames)
    with open(path, "wb") as f:
        f.write(video_bytes)

export_to_video(output, "output5.mp4", fps=15)
