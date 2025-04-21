#!/usr/bin/env python
# coding: utf-8

import torch
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel, BitsAndBytesConfig
import imageio as iio
import math
import numpy as np
import io
import time
import pandas as pd
import json
from random import sample
import gc
from tqdm import tqdm
from TTS.api import TTS
from moviepy.editor import AudioFileClip, concatenate_videoclips, ImageSequenceClip, VideoFileClip
from scipy.io.wavfile import write
import os
import ollama

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
    
def setup_environment():
    """Set up the environment by disabling telemetry."""
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
        model='qwen2.5:32b',
        prompt=f"You are the world's best social media video script writer. Take the following story and turn it into an original viral short form video voiceover script for TikTok. The script should take no more than two minutes to narrate and should have a hook at the beginning to catch the viewer's attention in the first 5 seconds that isn't cheesy. For each scene, write a very detailed scene description like in a movie screenplay and include any camera movements. The voiceover should convey the entire story completely on its own, don't rely on the image descriptions to tell the story. Only for the image description come up with appropriate first and last names of any characters and always use their full name for each scene also describe them physically for each scene. Here's the story: {inspiration_text}"
    )
    return script['response']

def convert_to_json(script):
    """
    Convert the refined script into a JSON format with scenes containing image descriptions and voiceovers.
    
    Args:
        script (str): The refined script text.
        
    Returns:
        dict: The script in JSON format.
    """
    json_output = ollama.generate(
        model='qwen2.5:32b',
        format='json',
        keep_alive=0,
        prompt=f"Take the following script and turn it into json format, it should be an array containing scenes, each scene should contain a imageDescription and voiceover field. In the voiceover string change any single quotes to double quotes. Here's the script: {script}"
    )
    return json.loads(json_output['response'])

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

def generate_hunyuan_video(prompt, output_path, fps=14, height=720, width=480, num_frames=120, prompt_template=prompt_template, num_inference_steps=15):
    """
    Generate a video using the HunyuanVideoPipeline based on the provided prompt.
    
    Args:
        prompt (str): The description of the scene to generate a video for.
        output_path (str): The path to save the generated video file.
        fps (int, optional): Frames per second. Defaults to 14.
        height (int, optional): Height of the video frames. Defaults to 720.
        width (int, optional): Width of the video frames. Defaults to 480.
        num_frames (int, optional): Number of frames in the video. Defaults to 120.
        prompt_template (dict, optional): The template for the prompt. Defaults to a predefined template.
        num_inference_steps (int, optional): Number of inference steps. Defaults to 15.
    """
    output = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        prompt_template=prompt_template,
        num_inference_steps=num_inference_steps
    ).frames[0]
    
    export_to_video(output, output_path, fps=fps)

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
    return tts.tts(voiceover, speaker_wav=speaker_wavs, language="en"), tts


def create_video_from_scripts(scriptdicts, output_filenames, audio_dicts, sample_rate=24000, fps=24):
    """
    Create videos from the scenes in script dictionaries by combining images and pre-generated audio.
    
    Args:
        scriptdicts (list): List of scripts in JSON format with scenes containing image descriptions and voiceovers.
        output_filenames (list): List of filenames to save the final videos.
        audio_dicts (list): List of dictionaries mapping scene indices to their corresponding synthesized audio data.
        sample_rate (int, optional): The audio sample rate. Defaults to 24000.
        fps (int, optional): The frames per second for the video. Defaults to 24.
    """
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
    
    for idx, scriptdict in enumerate(tqdm(scriptdicts, desc="Creating Videos", unit="script")):
        clips = []
        
        for scene_idx, scene in enumerate(tqdm(scriptdict['scenes'], desc=f"Script {idx + 1} Scenes", leave=False)):
            video_filename = f'temp_video_{idx}_{scene_idx}.mp4'
            generate_hunyuan_video(scene['imageDescription'], video_filename)
            
            audio_filename = f'temp_audio_{idx}_{scene_idx}.wav'
            audio_data = audio_dicts[idx][scene_idx]
            write(audio_filename, sample_rate, (np.array(audio_data) * 32767).astype(np.int16))  # Convert float to int16
            
            # Load the video frames using MoviePy
            video_clip = VideoFileClip(video_filename)
            
            audio_clip = AudioFileClip(audio_filename, fps=sample_rate)
            
            video_duration = video_clip.duration
            audio_duration = audio_clip.duration
            
            if video_duration < audio_duration:
                # Extend the video by looping
                video_clip = video_clip.loop(duration=audio_duration)
            elif video_duration > audio_duration:
                # Trim the video to match the audio duration
                video_clip = video_clip.subclip(0, audio_duration)
            
            final_clip = video_clip.set_audio(audio_clip)
            clips.append(final_clip)
        
        final_video = concatenate_videoclips(clips)
        final_video.write_videofile(output_filenames[idx], fps=fps)
        for scene_idx in range(len(scriptdict['scenes'])):
            video_filename = f'temp_video_{idx}_{scene_idx}.mp4'
            if os.path.exists(video_filename):
                os.remove(video_filename)

            audio_filename = f'temp_audio_{idx}_{scene_idx}.wav'
            if os.path.exists(audio_filename):
                os.remove(audio_filename)

def process_batch_of_scripts(batch_size=10):
    setup_environment()
    df = load_data('approved_stories_qwen.csv')
    story_indices = sample(range(len(df)), batch_size)

    scripts = []
    speaker_wavs = ["zainvoice.wav", "zainvoice2.wav", "zainvoice3.wav", "zainvoice4.wav", "zainvoice5.wav", "zainvoice6.wav", "zainvoice7.wav"]
    audio_dicts = []

    for storynum in tqdm(story_indices, desc="Generating Scripts", unit="script"):
        inspo = df.iloc[storynum].selftext
        script_generated = False
        attempts = 0

        while not script_generated and attempts < 3:
            script = generate_script(inspo)
            scriptdict = convert_to_json(script)

            valid = True
            for scene in scriptdict['scenes']:
                if 'imageDescription' not in scene or 'voiceover' not in scene:
                    # Also fail if the voiceover is empty
                    if scene['voiceover'] == "":
                        valid = False
                        break
                    valid = False
                    break

            if valid:
                script_generated = True
                print(json.dumps(scriptdict, indent=4))
                user_input = input("Approve the script? (y/n): ")
                if user_input.lower() != "y":
                    skip_to_next_story = input("Skip to the next story? (y/n): ")
                    if skip_to_next_story.lower() == "y":
                         # Select a new random story index from the DataFrame
                        story_indices.remove(storynum)
                        new_story_index = sample(range(len(df)), 1)[0]
                        while new_story_index in story_indices:
                            new_story_index = sample(range(len(df)), 1)[0]
                        story_indices.append(new_story_index)                            
                        break
                    else:
                        script_generated = False
            else:
                attempts += 1
                print(f"Script for story {storynum} is invalid. Retrying...")

        if script_generated:
            output_filename = f'videos/story_{storynum}_output_video_{time.strftime("%Y_%m_%d-%I_%M_%S_%p")}.mp4'
            scripts.append((scriptdict, output_filename))

            audio_data_dict = {}
            for scene_idx, scene in enumerate(tqdm(scriptdict['scenes'], desc=f"Script {len(scripts)} Scenes", leave=False)):
                print(scene['voiceover'])
                audio_data, tts_model = synthesize_speech(scene['voiceover'], speaker_wavs)
                audio_data_dict[scene_idx] = audio_data
                audio_dicts.append(audio_data_dict)

    output_filenames = [output_filename for _, output_filename in scripts]

    # Unload the TTS model from memory
    del tts_model
    torch.cuda.empty_cache()

    create_video_from_scripts([scriptdict for scriptdict, _ in scripts], output_filenames, audio_dicts)

def main():
    process_batch_of_scripts(batch_size=1)

if __name__ == "__main__":
    main()