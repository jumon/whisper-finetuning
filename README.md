# [Work In Progress] whisper-finetuning
This is a collection of scripts that can be used to fine-tune a Whisper model using <strong>time-aligned</strong> transcriptions and audio files.
Although there are already codes available for fine-tuning a Whisper model, such as the one provided by the Hugging Face transformers library (https://huggingface.co/blog/fine-tune-whisper), they only offer a way to fine-tune a model using transcripts <strong>without timestamps</strong>.
This makes it difficult to output timestamps along with the transcriptions.
This repository, however, provides scripts that allow you to fine-tune a Whisper model using time-aligned data, making it possible to output timestamps with the transcriptions.

## Setup
```
pip install -r requirements.txt
```
You may also need to install [ffmpeg](https://ffmpeg.org/) and [rust](https://www.rust-lang.org/) to run Whisper depending on your environment.
See the [instructions](https://github.com/openai/whisper#setup) in the Whisper repository for more details if you encounter any errors.

## Usage
### 1. Prepare your data
The scripts in this repository assume that you have a directory containing audio files and a corresponding directory containing transcripts in SRT (or VTT) format.
The SRT (or VTT) files should have the same name as the audio files.
Run the following command to generate a jsonl file that can be used as a training set for finetuning a model:
```
python create_data.py --audio-dir <path-to-training-audio-dir> --transcript-dir <path-to-training-transcript-dir> --language <language-of-your-data>  --output train.json
```
To generate a jsonl file for validation, run the following command:
```
python create_data.py --audio-dir <path-to-dev-audio-dir> --transcript-dir <path-to-dev-transcript-dir> --language <language-of-your-data>  --output dev.json
```
For all available options, see `python create_data.py --help`.

### 2. Finetune a model
You can finetune a model with the jsonl files generated in the previous step:
```
python run_finetuning.py --train-json <path-to-train.json> --dev-json <path-to-dev.json> --model <model-name>
```
For all available options, see `python run_finetuning.py --help`.

### 3. Transcribe audio files
You can transcribe audio files using the finetuned model by running the command:
```
python transcribe.py --audio-dir <path-to-audio-dir-to-transcribe> --save-dir <output-dir> --language <language-of-your-data> --model <path-to-finetuned-model>
```
Alternatively, you can use the original [whisper command](https://github.com/openai/whisper#command-line-usage) with the `--model <path-to-finetuned-model>` option to transcribe audio files using the finetuned model.

### 4. Calculate a metric
To calculate a metric such as Word Error Rate (WER), use the transcripts generated in the previous step along with the ground truth transcripts by running the command:
```
python calculate_metric.py --recognized-dir <path-to-recognized-transcript-dir> --transcript-dir <path-to-ground-truth-transcript-dir> --metric WER
```
For all available options, see `python calculate_metric.py --help`.
