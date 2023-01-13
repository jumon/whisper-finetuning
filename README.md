# [Work In Progress] whisper-finetuning
This is a collection of scripts that can be used to fine-tune a Whisper model using "time-aligned" transcriptions and audio files.
Although there are already codes available for fine-tuning a Whisper model, such as the one provided by the Hugging Face transformers library (https://huggingface.co/blog/fine-tune-whisper), they only offer a way to fine-tune a model using transcripts without timestamps.
This makes it difficult to output timestamps along with the transcriptions.
This repository, however, provides scripts that allow you to fine-tune a Whisper model using time-aligned data, making it possible to output timestamps with the transcriptions.

## Setup
Install [Whisper](https://github.com/openai/whisper) and its dependencies:
```
pip install git+https://github.com/openai/whisper.git
```
You may also need to install [ffmpeg](https://ffmpeg.org/) and [rust](https://www.rust-lang.org/) depending on your environment.
See the [instructions](https://github.com/openai/whisper#setup) in the Whisper repository for more details if you encounter any errors.

## Usage
### 1. Prepare your data
The scripts in this repository assume that you have a directory containing audio files and a corresponding directory containing transcripts in SRT format.
The SRT files should have the same name as the audio files.
Run the following command to generate a jsonl file that can be used as a training set for finetuning a model:
```
python create_data.py --audio <path-to-training-audio-dir> --transcript <path-to-training-transcript-dir> --language <language-of-your-data>  --output train.json
```
To generate a jsonl file for validation, run the following command:
```
python create_data.py --audio <path-to-dev-audio-dir> --transcript <path-to-dev-transcript-dir> --language <language-of-your-data>  --output dev.json
```
For all available options, see `python create_data.py --help`.

### 2. Finetune a model
You can finetune a model with the jsonl files generated in the previous step:
```
python finetune.py --train-json <path-to-train.json> --dev-json <path-to-dev.json> --model <model-name>
```
For all available options, see `python finetune.py --help`.
