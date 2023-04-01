# [Work In Progress] whisper-finetuning
This is a collection of scripts that can be used to fine-tune a Whisper model using <strong>time-aligned</strong> transcriptions and audio files.
Although there are already codes available for fine-tuning a Whisper model, such as the one provided by the Hugging Face transformers library (https://huggingface.co/blog/fine-tune-whisper), they only offer a way to fine-tune a model using transcripts <strong>without timestamps</strong>.
This makes it difficult to output timestamps along with the transcriptions.
This repository, however, provides scripts that allow you to fine-tune a Whisper model using time-aligned data, making it possible to output timestamps with the transcriptions.

## Setup
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
To install pytorch, you may need to follow the instructions [here](https://pytorch.org/get-started/locally/) depending on your environment.
You may also need to install [ffmpeg](https://ffmpeg.org/) and [rust](https://www.rust-lang.org/) to run Whisper.
See the [instructions](https://github.com/openai/whisper#setup) in the Whisper repository for more details if you encounter any errors.

### Windows 
You will need to install the `soundfile` python package.
```
pip install soundfile
```
On Windows, you also have to change the `num_workers` parameter in `dataloader.py` (almost at the end of the file) by setting it to 0.

For using the Adam 8bit optimizer with the `bitsandbytes` package, you will need to download pre-built binaries from another repo, since by default it is not supported.
You can grab the .dll from [here](https://github.com/DeXtmL/bitsandbytes-win-prebuilt) or more easily download [this folder](https://github.com/bmaltais/kohya_ss/tree/master/bitsandbytes_windows) with the .dll and .py to patch and copy them using the following commands:
```
cp .\bitsandbytes_windows\*.dll .\venv\Lib\site-packages\bitsandbytes\
cp .\bitsandbytes_windows\cextension.py .\venv\Lib\site-packages\bitsandbytes\cextension.py
cp .\bitsandbytes_windows\main.py .\venv\Lib\site-packages\bitsandbytes\cuda_setup\main.py
```

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
You can use the `--use-adam-8bit` flag to utilize the Adam 8bit optimizer from `bitsandbytes`. This will reduce VRAM usage and allows to train using small multimodal models with 8GB of VRAM.

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
