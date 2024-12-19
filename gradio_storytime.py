#!/usr/bin/env python

from IPython import get_ipython

from moviepy.editor import ImageClip, concatenate_videoclips, AudioFileClip
from scipy.io.wavfile import write
import numpy as np
import os

import ollama
import pandas as pd
import json

import time

from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
import torch

import gc
import torch
from TTS.api import TTS

import gradio as gr


# !pip install pandas
# !pip install ollama
#!pip install 'diffusers[torch]' transformers
#!pip install TTS
#!pip install moviepy

os.environ['DISABLE_TELEMETRY'] = 'YES'


df = pd.read_csv('top1k.csv')





def generate_script_from_story(storynum):
    inspo = df.iloc[storynum].selftext
    print(inspo)

    script = ollama.generate(model='llama3:70b-instruct', 
                            prompt='''You are the world's best social media video script writer. Take the following story and turn it into an original viral short form video voiceover script for tiktok.
    The script should take no more than two minutes to narrate and should have a hook at the beginning to catch the viewer's attention in the first 5 seconds that isn't cheesy.
    For each scene, write a very detailed description of the scene in a way that could be used by a stable diffusion ai model
    to generate an image to accompany the voiceover.Only for the image description come up with appropriate first and last names of any characters and always use their full name for each scene also describe them physically for each scene. Here's the story: ''' + inspo)

    print(script['response'])

    with open('promptguide.txt', 'r') as file:
        promptguide = file.read()

    scriptwdesc = ollama.generate(model='llama3:70b-instruct', 
                            prompt='''Take the image descriptions in the script and change them on using the techniques in the prompt guide i have attached below the script. the output should be the edited script (voiceovers and descriptions).
    copy the voiceovers as is for each scene and ignore any afterword after the script. DO NOT DESCRIBE WHAT THE GUIDE DOES OR SUMMARIZE THE GUIDE!!! INCLUDE THE VOICEOVER!
    ''' +
    '''Here's the prompt guide: ''' + promptguide + ' heres the script you need to edit: Script:' + script['response'] )

    #print(scriptwdesc['response'])
    return scriptwdesc['response']

def generate_images(scriptwdesc):
    scriptjson = ollama.generate(model='llama3:70b-instruct', 
                                format='json',
                                keep_alive=1,
                            prompt='''Take the following script and turn it into json format, it should be an array containing scenes, each scene should contain a imageDescription and voiceover field. in the voiceover string change any single quotes to double quotes.
                            here's the script: ''' + scriptwdesc)


    print(scriptjson['response'])

    scriptdict = json.loads(scriptjson['response'])

    time.sleep(30)

    pipeline = StableDiffusionXLPipeline.from_single_file(
        "Juggernaut_X_RunDiffusion.safetensors",
        torch_dtype=torch.float16, variant="fp16")
    pipeline = pipeline.to("cuda")
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)


    #prompt = "High resolution Portrait of a stylish African woman in urban setting, short brunette hair, bold red lipstick, colors striking red and deep blacks, style modern fashion, mood confident, lighting high contrast with sharp shadows, perspective frontal view, texture leather jacket and smooth skin"
    def make_image(prompt):
        images = pipeline(prompt=prompt,
            negative_prompt=" nudity, (worst quality, low quality, normal quality, lowres, low details, oversaturated, undersaturated, overexposed, underexposed, grayscale, bw, bad photo, bad photography, bad art:1.4), (watermark, signature, text font, username, error, logo, words, letters, digits, autograph, trademark, name:1.2), (blur, blurry, grainy), morbid, ugly, asymmetrical, mutated malformed, mutilated, poorly lit, bad shadow, draft, cropped, out of frame, cut off, censored, jpeg artifacts, out of focus, glitch, duplicate, (airbrushed, cartoon, anime, semi-realistic, cgi, render, blender, digital art, manga, amateur:1.3), (3D ,3D Game, 3D Game Scene, 3D Character:1.1), (bad hands, bad anatomy, bad body, bad face, bad teeth, bad arms, bad legs, deformities:1.3)",
            active_tags=[],
            inactive_tags=[],
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
            lora_alpha=[],
            #num_outputs=8,
            output_format="png").images[0]
        return images


    for scene in scriptdict['scenes']:
        print(scene['imageDescription'])
        scene['image'] = make_image(scene['imageDescription'])
        #display(scene['image'])
        

    del pipeline

    def flush():
        gc.collect()
        torch.cuda.empty_cache()
        
    flush()
    return scriptdict

def generate_voiceover(scriptdict):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)


    for scene in scriptdict['scenes']:
        print(scene['voiceover'])
        scene['wav'] = tts.tts(scene['voiceover'], speaker_wav=["zainvoice.wav","zainvoice2.wav","zainvoice3.wav","zainvoice4.wav","zainvoice5.wav","zainvoice6.wav","zainvoice7.wav"], language="en")



    def create_video_from_scripts(scriptdict, output_filename, sample_rate=24000, fps=24):
        clips = []
        
        for idx, scene in enumerate(scriptdict['scenes']):
            # Convert the PIL image to an array
            img_array = np.array(scene['image'])
            
            # Create an ImageClip from the image array
            image_clip = ImageClip(img_array)
            
            # Save the audio array to a temporary WAV file
            audio_array = scene['wav']
            audio_filename = f'temp_audio_{idx}.wav'
            write(audio_filename, sample_rate, np.array(audio_array))
            
            # Create an AudioFileClip from the WAV file
            audio_clip = AudioFileClip(audio_filename, fps=24000)
            
            # Set the duration of the image clip to match the duration of the audio clip
            image_clip = image_clip.set_duration(audio_clip.duration)
            
            # Set the audio of the image clip
            image_clip = image_clip.set_audio(audio_clip)
            image_clip.fps = 24
            
            clips.append(image_clip)
        
        # Concatenate all the clips
        final_clip = concatenate_videoclips(clips)
        # Write the final video file
        final_clip.write_videofile(output_filename)
        
        # Clean up temporary audio files
        for idx in range(len(scriptdict['scenes'])):
            audio_filename = f'temp_audio_{idx}.wav'
            if os.path.exists(audio_filename):
                os.remove(audio_filename)

    create_video_from_scripts(scriptdict, 'videos/story_'+str(storynum)+'_output_video_'+ time.strftime("%Y_%m_%d-%I_%M_%S_%p")+'.mp4')

    # del pipeline

    # def flush():
    #     gc.collect()
    #     torch.cuda.empty_cache()

    # flush()


#generate videos in a loop:
#from random import randint
# goodinspo = []
# for storynum in range(20, 50):
#     try:
#         generate_vid_from_story(storynum)
#     except:
#         print('THERE WAS A PROBLEM BRO!---------------------------------')
#         continue

demo = gr.Blocks()

with demo:
    audio_file = gr.Audio(type="filepath")
    text = gr.Textbox()
    label = gr.Label()

    b1 = gr.Button("Recognize Speech")
    b2 = gr.Button("Classify Sentiment")

    b1.click(speech_to_text, inputs=audio_file, outputs=text)
    b2.click(text_to_sentiment, inputs=text, outputs=label)

demo.launch()