import argparse
from pathlib import Path
from typing import Iterator, Union

import torch
import whisper
from tqdm import tqdm
from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE
from whisper.utils import get_writer


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transcribe audio files with a Whisper model")
    parser.add_argument(
        "--audio-dir",
        type=str,
        required=True,
        help="Path to directory containing audio files to transcribe",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="output",
        help="Path to directory to save transcribed results",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        choices=sorted(LANGUAGES.keys()) + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]),
        help=(
            "Language of the data. The corresponding language tag will be used as an input to the "
            "decoder of the Whisper model."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for PyTorch inference",
    )
    parser.add_argument("--model", default="large", help="Name or path to the Whisper model to use")
    parser.add_argument(
        "--task",
        type=str,
        default="transcribe",
        choices=["transcribe", "translate"],
        help=(
            "Whether to perform X->X speech recognition ('transcribe')"
            "or X->English translation ('translate')"
        ),
    )
    return parser


def main():
    args = get_parser().parse_args()
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    model = whisper.load_model(args.model, args.device)
    writer = get_writer("srt", args.save_dir)

    for audio_path in tqdm(list(Path(args.audio_dir).iterdir())):
        result = model.transcribe(task=args.task, audio=str(audio_path), language=args.language)
        writer(result, str(audio_path))


if __name__ == "__main__":
    main()
