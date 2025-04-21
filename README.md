# AI Social Media Reddit Story Video Generator

## Overview

This project automates the creation of engaging, short-form social media videos (like TikToks or Reels) based on popular Reddit stories. It leverages a multi-stage pipeline involving cutting-edge Generative AI models for scriptwriting, voice narration, and video generation, culminating in a final, edited video ready for posting.

The goal is to transform text-based narratives into compelling visual stories, optimized for capturing attention and driving engagement on social platforms. This tool demonstrates an end-to-end workflow for automated AI-driven content creation.

## Project Goal & Relevance

This project addresses the growing need for scalable, AI-powered content creation tools for social media management. It's particularly relevant for platforms aiming to leverage AI for content generation, especially involving image and video.

It showcases hands-on experience in:

* **Building Next-Generation Generative Models:** Implementing and orchestrating multiple state-of-the-art generative AI models (LLM for text, TTS for audio, Text-to-Video) in a cohesive pipeline.
* **Optimizing for Social Media:** The script generation prompt is specifically designed to create viral-style hooks and narratives suitable for platforms like TikTok, demonstrating an understanding of content optimization for **high engagement**.
* **End-to-End System Design & Productionization:** Architecting a complete workflow from raw text input (Reddit stories) to a finished video product (.mp4), including handling dependencies, model interactions, intermediate file management, and error handling/retries. This reflects experience in building scalable and robust AI solutions.
* **Hands-On Generative AI Skills:** Practical application of text-to-video models (`tencent/HunyuanVideo`) and large language models (`qwen2.5:32b` via Ollama) for creative generation.
* **Optimization Techniques:** Utilizes 4-bit quantization (`BitsAndBytesConfig`) for the HunyuanVideo transformer, demonstrating awareness of techniques for **optimizing inference pipelines** for speed and **cost efficiency** – crucial for scaling AI features.
* **Independent Project Execution:** The development of this complex, multi-stage project demonstrates the ability to independently drive technical projects from concept to completion.

## Key Features

* **Automated Script Generation:** Uses the Qwen 2.5 32B LLM via Ollama to generate engaging video scripts with detailed scene descriptions and voiceover text from Reddit stories.
* **AI Voice Narration:** Employs Coqui TTS (XTTSv2 model) for high-quality, multi-lingual text-to-speech synthesis, bringing the script to life.
* **AI Video Generation:** Leverages the `tencent/HunyuanVideo` text-to-video diffusion model via the `diffusers` library to create video clips corresponding to scene descriptions.
* **Efficient Inference:** Implements 4-bit quantization for the video model transformer to reduce memory footprint and potentially speed up inference.
* **Automated Video Editing:** Uses `moviepy` to dynamically adjust video clip lengths to match audio duration and concatenates scenes into a final video file.
* **Batch Processing:** Capable of processing multiple stories in a batch.
* **User Validation:** Includes an optional step for manual script approval before proceeding with resource-intensive generation steps.

## Workflow Breakdown

```
+---------------------+      +----------------------+      +-------------------------+
| 1. Data Loading     |----->| 2. Script Generation   |----->| 3. Script Structuring   |
| (reads CSV:         |      | (Qwen LLM:            |      | (Qwen LLM:             |
| approved_stories_   |      | generates script with |      | formats script to JSON  |
| qwen.csv - Reddit   |      | visual & voiceover)   |      | - 'scenes' array with   |
| story text)         |      | (ollama.generate)     |      | 'imageDescription' &    |
+---------------------+      +----------------------+      | 'voiceover' fields)     |
                                                           | (ollama.generate,       |
                                                           | format='json')          |
                                                           +-------------------------+
                                                                      |
                                                                      v
                                                    +---------------------------------+
                                                    | 4. Script Validation & Approval |
                                                    | (Optional: User reviews JSON,  |
                                                    | approves, retries, or skips)    |
                                                    +---------------------------------+
                                                                      |
                                                                      v
+---------------------------------+      +-------------------------------------+      +-----------------------------------+
| 5. Audio Generation (Scene-by-  |----->| 6. Video Generation (Scene-by-Scene)|----->| 7. Video & Audio Editing (Scene-by-|
| Scene)                          |      | (HunyuanVideo Pipeline:            |      | Scene)                            |
| (Coqui TTS: voiceover text ->   |      | imageDescription -> video clip)    |      | (moviepy: load audio & video,      |
| audio waveform using            |      | (diffusers.HunyuanVideoPipeline)    |      | adjust video duration, attach audio)|
| speaker_wavs)                   |      |                                     |      +-----------------------------------+
| (TTS.api)                       |      |                                     |                      |
+---------------------------------+      +-------------------------------------+                      |
                                                                                                   |
                                                                                                   v
                                                                           +-------------------------+
                                                                           | 8. Output & Cleanup     |
                                                                           | (Save final video,     |
                                                                           | delete temp files)      |
                                                                           +-------------------------+
```

1.  **Data Loading:** Reads Reddit story text (e.g., `selftext`) from a CSV file (`approved_stories_qwen.csv`).
2.  **Script Generation:**
    * A randomly selected Reddit story is fed to the Qwen LLM (`ollama.generate`).
    * The LLM generates a short-form video script, including scene-by-scene visual descriptions and voiceover text, prompted to create a viral hook and detailed imagery.
3.  **Script Structuring:**
    * The generated script text is passed back to the Qwen LLM with a prompt to format it into a structured JSON array (`scenes`) where each element contains `imageDescription` and `voiceover` fields (`ollama.generate` with `format='json'`).
4.  **Script Validation & Approval (Optional):**
    * The script's JSON structure is validated.
    * The user is prompted to approve the generated script before proceeding. Allows for retries or skipping to the next story.
5.  **Audio Generation (Scene-by-Scene):**
    * For each approved scene, the `voiceover` text is fed to the Coqui TTS model (`TTS.api`).
    * An audio waveform is generated using pre-defined speaker voice samples (`speaker_wavs`).
    * The generated audio data is stored temporarily.
6.  **Video Generation (Scene-by-Scene):**
    * For each scene, the `imageDescription` is used as a prompt for the HunyuanVideo pipeline (`diffusers.HunyuanVideoPipeline`).
    * A short video clip (.mp4) is generated based on the description and saved temporarily.
7.  **Video & Audio Editing:**
    * For each scene:
        * The temporary audio and video files are loaded using `moviepy`.
        * The video clip duration is adjusted (looped or trimmed) to precisely match the audio duration using `VideoFileClip.loop` or `VideoFileClip.subclip`.
        * The audio is attached to the adjusted video clip (`VideoFileClip.set_audio`).
    * All processed scene clips are concatenated into a single final video (`moviepy.editor.concatenate_videoclips`).
8.  **Output & Cleanup:**
    * The final video is saved to the specified output path (e.g., `videos/story_{id}_output_video_{timestamp}.mp4`).
    * Temporary audio and video files for each scene are deleted.

## Technical Details

### Core Technologies

* **Python 3.x**
* **PyTorch:** Deep learning framework.
* **Diffusers (Hugging Face):** Library for diffusion models, used here for `HunyuanVideoPipeline`.
* **Transformers (Hugging Face):** Used implicitly by Diffusers.
* **BitsAndBytes:** For model quantization (4-bit).
* **Ollama:** For running local LLMs (Qwen 2.5 32B).
* **Coqui TTS:** Text-to-speech library (`tts_models/multilingual/multi-dataset/xtts_v2`).
* **MoviePy:** Video editing library.
* **NumPy:** Numerical operations.
* **Pandas:** Data manipulation (for reading CSV).
* **ImageIO:** Used by `moviepy` and for potential frame manipulation (though `moviepy` handles most direct video I/O here).
* **SciPy:** Used for writing WAV files (`scipy.io.wavfile.write`).

### Setup & Installation

1.  **Clone the Repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```
2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate # Linux/macOS
    # venv\Scripts\activate # Windows
    ```
3.  **Install Python Dependencies:**
    ```bash
    pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118) # Adjust CUDA version if needed
    pip install diffusers transformers accelerate bitsandbytes TTS moviepy pandas numpy imageio scipy ollama tqdm
    ```
    *Note: Ensure you have a compatible CUDA setup if using GPU acceleration.*
4.  **Install Ollama:** Follow the instructions on the [Ollama website](https://ollama.com/) to install Ollama for your operating system.
5.  **Pull the Qwen Model:**
    ```bash
    ollama pull qwen2.5:32b
    ```
6.  **Prepare Input Data:**
    * Create a CSV file named `approved_stories_qwen.csv` in the project's root directory.
    * It should contain at least one column named `selftext` holding the Reddit stories you want to process.
    * Example `approved_stories_qwen.csv`:
        ```csv
        id,title,selftext
        1,"Story Title 1","This is the first Reddit story text..."
        2,"Story Title 2","This is the second Reddit story text..."
        ```
7.  **Prepare Speaker WAV Files:**
    * Place one or more high-quality WAV audio files to be used as voice references for Coqui TTS in the project's root directory.
    * Update the `speaker_wavs` list in the `process_batch_of_scripts` function in the Python script to match your filenames (e.g., `speaker_wavs = ["my_voice.wav"]`).
8.  **Create Output Directory:**
    ```bash
    mkdir videos
    ```

### Configuration

* **Models:** Model IDs (`tencent/HunyuanVideo`, `qwen2.5:32b`, `tts_models/multilingual/multi-dataset/xtts_v2`) are hardcoded but can be modified.
* **Quantization:** `BitsAndBytesConfig` is used for 4-bit quantization of the HunyuanVideo transformer.
* **Video Parameters:** `fps`, `height`, `width`, `num_frames`, `num_inference_steps` can be adjusted in the `generate_hunyuan_video` function call within `create_video_from_scripts`.
* **Batch Size:** Controlled by the `batch_size` parameter in the `main` function's call to `process_batch_of_scripts`.
* **Speaker Voices:** Modify the `speaker_wavs` list in `process_batch_of_scripts`.

### Running the Script

Execute the main script from your terminal:

```bash
python your_script_name.py # Replace your_script_name.py with the actual filename
```
The script will then:

    Load a story from the CSV.

    Generate and potentially ask for approval of the script.

    Generate audio and video for each scene (this can take significant time and compute resources, especially GPU).

    Edit the scenes together.

    Save the final video to the videos/ directory.

    Clean up temporary files.

Future Enhancements

    Integration with Reddit API (PRAW) for automatic story fetching.

    More sophisticated video editing (transitions, effects, text overlays).

    Support for other LLMs, TTS, or Video Generation models.

    Improved error handling and logging.
