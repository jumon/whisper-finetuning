import argparse
from pathlib import Path
from typing import Iterator, Union

import evaluate
import torch
import whisper
from tqdm import tqdm
from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE
from whisper.utils import write_srt

from create_data import read_utterances_from_srt


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcribe audio files with a Whisper model and calculate evaluation metrics"
    )
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to directory containing audio files to transcribe",
    )
    parser.add_argument(
        "--transcript",
        type=str,
        default=None,
        help=(
            "Path to directory containing transcripts in SRT format. Defaults to None, in which "
            "case generated transcripts will not be evaluated. When this argument is set, the "
            "filenames under this directory must match the filenames under `--audio` directory "
            "except for the extension. For example, if the transcript file is `example.srt`, there "
            "must be an audio file like `example.wav` under `--audio` directory."
        ),
    )
    parser.add_argument(
        "--output", type=str, default="output", help="Path to directory to save transcribed results"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        choices=sorted(LANGUAGES.keys()) + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]),
        help="Language of the data",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="WER",
        choices=["WER", "CER"],
        help="Evaluation metric. This option is used only when `--transcript` is set.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print out the evaluation results of each file"
    )

    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="device to use for PyTorch inference",
    )
    parser.add_argument("--model", default="large", help="name or path to the Whisper model to use")
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


def srt_to_text(path: Union[str, Path]) -> str:
    utterances = read_utterances_from_srt(path, "", normalize_unicode=True)
    return " ".join([u.text for u in utterances])


def save_srt(transcript: Iterator[dict], path: Union[str, Path]) -> None:
    with open(path, "w") as f:
        write_srt(transcript, file=f)


def main():
    args = get_parser().parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    model = whisper.load_model(args.model, args.device)
    do_evaluation = args.transcript is not None

    if do_evaluation:
        reference_texts, recognized_texts = [], []
        evaluator = evaluate.load(args.metric.lower())
        score_sum = 0

    for audio_path in tqdm(list(Path(args.audio).iterdir())):
        if args.verbose:
            tqdm.write(f"Processing: {audio_path}")

        speech_id = Path(audio_path).stem
        result = model.transcribe(task=args.task, audio=str(audio_path), language=args.language)
        recognized_path = Path(args.output) / f"{speech_id}.srt"
        save_srt(result["segments"], recognized_path)

        if do_evaluation:
            transcript_path = Path(args.transcript) / f"{speech_id}.srt"
            if not transcript_path.exists():
                raise FileNotFoundError(f"Transcript file not found: {transcript_path}")

            reference_text = srt_to_text(transcript_path)
            recognized_text = srt_to_text(recognized_path)
            reference_texts.append(reference_text)
            recognized_texts.append(recognized_text)

            score = evaluator.compute(references=[reference_text], predictions=[recognized_text])
            if args.verbose:
                tqdm.write(f"    {args.metric}: {score}")
            score_sum += score

    if do_evaluation:
        print(f"Unweighted Average {args.metric}: {score_sum / len(reference_texts)}")
        weighted_average = evaluator.compute(
            references=reference_texts, predictions=recognized_texts
        )
        print(f"Weighted Average {args.metric}: {weighted_average}")


if __name__ == "__main__":
    main()
