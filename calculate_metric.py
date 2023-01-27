import argparse
from pathlib import Path
from typing import Union

import evaluate
from tqdm import tqdm

from create_data import DataProcessor


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calculate evaluation metrics")
    parser.add_argument(
        "--recognized-dir",
        type=str,
        required=True,
        help="Path to directory containing recognized transcripts in SRT format",
    )
    parser.add_argument(
        "--transcript-dir",
        type=str,
        required=True,
        help=(
            "Path to directory containing transcripts in SRT (or VTT) format. The filenames under "
            "this directory must match the filenames under `--recognized-dir` directory."
        ),
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="WER",
        choices=["WER", "CER"],
        help="Evaluation metric",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print out the evaluation results of each file"
    )
    return parser


def srt_to_text(path: Union[str, Path], utterance_separator: str = " ") -> str:
    utterances = DataProcessor.read_utterances_from_srt(path, normalize_unicode=True)
    return utterance_separator.join([u.text for u in utterances])


def vtt_to_text(path: Union[str, Path], utterance_separator: str = " ") -> str:
    utterances = DataProcessor.read_utterances_from_vtt(path, normalize_unicode=True)
    return utterance_separator.join([u.text for u in utterances])


def main():
    args = get_parser().parse_args()

    reference_texts, recognized_texts = [], []
    evaluator = evaluate.load(args.metric.lower())
    score_sum = 0
    utterance_separator = " " if args.metric == "WER" else ""

    for recognized_path in tqdm(list(Path(args.recognized_dir).iterdir())):
        speech_id = Path(recognized_path).stem

        if (Path(args.transcript_dir) / f"{speech_id}.srt").exists():
            transcript_path = Path(args.transcript_dir) / f"{speech_id}.srt"
            reference_text = srt_to_text(transcript_path, utterance_separator=utterance_separator)
        elif (Path(args.transcript_dir) / f"{speech_id}.vtt").exists():
            transcript_path = Path(args.transcript_dir) / f"{speech_id}.vtt"
            reference_text = vtt_to_text(transcript_path, utterance_separator=utterance_separator)
        else:
            raise FileNotFoundError(f"Transcript file not found for {speech_id}")

        recognized_text = srt_to_text(recognized_path, utterance_separator=utterance_separator)
        reference_texts.append(reference_text)
        recognized_texts.append(recognized_text)

        score = evaluator.compute(references=[reference_text], predictions=[recognized_text])
        if args.verbose:
            tqdm.write(f"Processing: {recognized_path}")
            tqdm.write(f"    {args.metric}: {score}")
        score_sum += score

    print(f"Unweighted Average {args.metric}: {score_sum / len(reference_texts)}")
    weighted_average = evaluator.compute(references=reference_texts, predictions=recognized_texts)
    print(f"Weighted Average {args.metric}: {weighted_average}")


if __name__ == "__main__":
    main()
