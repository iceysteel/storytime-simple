#!/usr/bin/env python
# coding: utf-8

import ollama
import pandas as pd
import json
from random import sample, randint
import time
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler, StableVideoDiffusionPipeline
import torch
import cv2
from TTS.api import TTS
from moviepy.editor import ImageSequenceClip, AudioFileClip, concatenate_videoclips
from scipy.io.wavfile import write
import numpy as np
import os
import gc
from tqdm import tqdm

def setup_environment():
    """Set up the environment by disabling telemetry."""
    # get_ipython().system('export DISABLE_TELEMETRY=YES')
    os.environ['DISABLE_TELEMETRY'] = 'YES'

def load_data(file_path):
    """Load data from a CSV file into a pandas DataFrame."""
    return pd.read_csv(file_path)

def generate_script(inspiration_text):
    """
    Generate a video script based on the provided inspiration text.
    
    Args:
        inspiration_text (str): The story or text to inspire the script.
        
    Returns:
        str: The generated script.
    """
    script = ollama.generate(
        model='llama3:70b-instruct',
        prompt=f"You are the world's best social media video script writer. Take the following story and turn it into an original viral short form video voiceover script for TikTok. The script should take no more than two minutes to narrate and should have a hook at the beginning to catch the viewer's attention in the first 5 seconds that isn't cheesy. For each scene, write a very detailed description of the scene in a way that could be used by a stable diffusion AI model to generate an image to accompany the voiceover. The voiceover should convey the entire story completely on its own, don't rely on the image descriptions to tell the story. Only for the image description come up with appropriate first and last names of any characters and always use their full name for each scene also describe them physically for each scene. Here's the story: {inspiration_text}"
    )
    return script['response']

def refine_script(script, promptguide):
    """
    Refine the generated script using a prompt guide to improve image descriptions.
    
    Args:
        script (str): The original script.
        promptguide (str): The prompt guide text for refining the script.
        
    Returns:
        str: The refined script.
    """
    refined_script = ollama.generate(
        model='llama3:70b-instruct',
        prompt=f"Take the image descriptions and the voiceovers in the script and change the image descriptions using the techniques in the prompt guide i have attached below the script. The output should be the edited script (voiceovers and descriptions). Copy the voiceovers as is for each scene and ignore any afterword after the script. DON'T FORGET TO INCLUDE THE ORIGINAL VOICEOVER FROM THE SCRIPT! Here's the prompt guide: {promptguide} heres the script where you need to edit the image descriptions but leave the voiceover the same: Script: {script}"
    )
    return refined_script['response']

def convert_to_json(script):
    """
    Convert the refined script into a JSON format with scenes containing image descriptions and voiceovers.
    
    Args:
        script (str): The refined script text.
        
    Returns:
        dict: The script in JSON format.
    """
    json_output = ollama.generate(
        model='llama3:70b-instruct',
        format='json',
        keep_alive=1,
        prompt=f"Take the following script and turn it into json format, it should be an array containing scenes, each scene should contain a imageDescription and voiceover field. In the voiceover string change any single quotes to double quotes. Here's the script: {script}"
    )
    return json.loads(json_output['response'])

def make_image(prompt):
    """
    Generate an image using the Stable Diffusion XL pipeline based on the provided prompt.
    
    Args:
        prompt (str): The description of the scene to generate an image for.
        
    Returns:
        PIL.Image: The generated image.
    """
    pipeline = StableDiffusionXLPipeline.from_single_file(
        "Juggernaut_X_RunDiffusion.safetensors",
        torch_dtype=torch.float16, variant="fp16"
    )
    pipeline = pipeline.to("cuda")
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    
    image = pipeline(prompt=f"(anime, semi-realistic, cgi, render, blender, digital art, manga, amateur:1.3), (3D ,3D Game, 3D Game Scene, 3D Character:1.1), {prompt}",
        negative_prompt="naked, penis, pussy, porn, nudity, (worst quality, low quality, normal quality, lowres, low details, oversaturated, undersaturated, overexposed, underexposed, grayscale, bw, bad photo, bad photography, bad art:1.4), (watermark, signature, text font, username, error, logo, words, letters, digits, autograph, trademark, name:1.2), (blur, blurry, grainy), morbid, ugly, asymmetrical, mutated malformed, mutilated, poorly lit, bad shadow, draft, cropped, out of frame, cut off, censored, jpeg artifacts, out of focus, glitch, duplicate, (bad hands, bad anatomy, bad body, bad face, bad teeth, bad arms, bad legs, deformities:1.3)",
        width=832,
        height=1216,
        num_inference_steps=30,
        guidance_scale=7.0,
        preserve_init_image_color_profile=False,
        upscale_amount=4,
        latent_upscaler_steps=10,
        sampler_name="dpmpp_2m_sde",
        clip_skip=True,
        tiling="none",
        use_vae_model="",
        use_controlnet_model="",
        control_filter_to_apply="",
        use_lora_model=[],
        lora_alpha=[]
    ).images[0]
    return image

def make_video(image):
    """
    Generate a video sequence from the provided image using the Stable Video Diffusion pipeline.
    
    Args:
        image (PIL.Image): The input image to generate frames from.
        
    Returns:
        list: A list of PIL images representing the video frames.
    """
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
    )
    pipe.enable_model_cpu_offload()
    
    generator = torch.manual_seed(42)
    frames = pipe(image, decode_chunk_size=8, generator=generator, motion_bucket_id=30, noise_aug_strength=0.1).frames[0]
    fixed_frames = list(map(lambda frame: frame.resize((832, 1216)), frames))
    
    return fixed_frames

def flush_memory():
    """Free up memory by deleting pipeline objects and clearing GPU cache."""
    if 'pipeline' in globals():
        del pipeline
    if 'pipe' in globals():
        del pipe
    gc.collect()
    torch.cuda.empty_cache()

def synthesize_speech(voiceover, speaker_wavs):
    """
    Synthesize speech for the given voiceover using the specified speaker WAV files.
    
    Args:
        voiceover (str): The text to be converted into speech.
        speaker_wavs (list): List of paths to speaker WAV files.
        
    Returns:
        numpy.ndarray: The synthesized audio data.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    return tts.tts(voiceover, speaker_wav=speaker_wavs, language="en")

def create_video_from_scripts(scriptdicts, output_filenames, sample_rate=24000, fps=24):
    """
    Create videos from the scenes in script dictionaries by combining images and audio.
    
    Args:
        scriptdicts (list): List of scripts in JSON format with scenes containing image descriptions and voiceovers.
        output_filenames (list): List of filenames to save the final videos.
        sample_rate (int, optional): The audio sample rate. Defaults to 24000.
        fps (int, optional): The frames per second for the video. Defaults to 24.
    """
    for idx, scriptdict in enumerate(tqdm(scriptdicts, desc="Creating Videos", unit="script")):
        clips = []
        
        for scene in tqdm(scriptdict['scenes'], desc=f"Script {idx + 1} Scenes", leave=False):
            img_arrays = list(map(np.array, scene['videoframes']))
            video_clip = ImageSequenceClip(sequence=img_arrays, fps=fps)
            
            audio_filename = f'temp_audio_{idx}.wav'
            write(audio_filename, sample_rate, np.array(scene['wav']))
            audio_clip = AudioFileClip(audio_filename, fps=sample_rate)
            
            video_clip = video_clip.loop()
            video_clip = video_clip.set_duration(audio_clip.duration)
            video_clip = video_clip.set_audio(audio_clip)
            
            clips.append(video_clip)
        
        final_clip = concatenate_videoclips(clips)
        final_clip.write_videofile(output_filenames[idx])
        
        for scene_idx in range(len(scriptdict['scenes'])):
            audio_filename = f'temp_audio_{scene_idx}.wav'
            if os.path.exists(audio_filename):
                os.remove(audio_filename)

def process_batch_of_scripts(batch_size=10):
    setup_environment()
    df = load_data('approved_stories.csv')
    story_indices = sample(range(len(df)), batch_size)
    
    with open('promptguide.txt', 'r') as file:
        promptguide = file.read()

    scripts = []
    for storynum in tqdm(story_indices, desc="Generating Scripts", unit="script"):
        inspo = df.iloc[storynum].selftext
        script = generate_script(inspo)
        refined_script = refine_script(script, promptguide)
        scriptdict = convert_to_json(refined_script)
        scripts.append((scriptdict, f'videos/story_{storynum}_output_video_{time.strftime("%Y_%m_%d-%I_%M_%S_%p")}.mp4'))

    for idx in tqdm(range(len(scripts)), desc="Generating Images and Videos", unit="script"):
        scriptdict, _ = scripts[idx]
        
        for scene in tqdm(scriptdict['scenes'], desc=f"Script {idx + 1} Scenes", leave=False):
            print(scene['imageDescription'])
            scene['image'] = make_image(scene['imageDescription'])
            scene['videoframes'] = make_video(scene['image'])

    flush_memory()

    speaker_wavs = ["zainvoice.wav", "zainvoice2.wav", "zainvoice3.wav", "zainvoice4.wav", "zainvoice5.wav", "zainvoice6.wav", "zainvoice7.wav"]
    for idx in tqdm(range(len(scripts)), desc="Synthesizing Speech", unit="script"):
        scriptdict, _ = scripts[idx]
        
        for scene in tqdm(scriptdict['scenes'], desc=f"Script {idx + 1} Scenes", leave=False):
            print(scene['voiceover'])
            scene['wav'] = synthesize_speech(scene['voiceover'], speaker_wavs)

    flush_memory()

    output_filenames = [output_filename for _, output_filename in scripts]
    create_video_from_scripts([scriptdict for scriptdict, _ in scripts], output_filenames)

def main():
    process_batch_of_scripts(batch_size=1)

if __name__ == "__main__":
    main()