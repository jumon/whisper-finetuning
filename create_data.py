import argparse
import json
import unicodedata
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List, Optional, Union

import torch
import torchaudio
from whisper.audio import load_audio
from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE, Tokenizer, get_tokenizer
from whisper.utils import format_timestamp


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a jsonl file to be used for fine-tuning a Whisper model"
    )
    parser.add_argument(
        "--with-timestamps",
        action="store_true",
        help=(
            "Read SRT files and audio files to create a jsonl file with timestamps and prompts for "
            "fine-tuning a Whisper model with time-aligned data"
        ),
    )
    parser.add_argument(
        "--without-timestamps",
        action="store_false",
        dest="with-timestamps",
        help=(
            "Read a text file containing audio filenames and transcriptions to create a jsonl file "
            "without timestamps and prompts. This will be used for fine-tuning a Whisper model "
            "with utterance-by-utterance data"
        ),
    )
    parser.set_defaults(with_timestamps=True)

    parser.add_argument(
        "--audio",
        type=str,
        help=(
            "Path to directory containing audio files. This option is used only when "
            "`--with-timestamps` is set. Audio formats that can be read by ffmpeg are supported."
        ),
    )
    parser.add_argument(
        "--transcript",
        type=str,
        help=(
            "Path to directory containing transcripts in SRT format. This option is used only "
            "when `--with-timestamps` is set. Filenames must match the filenames under `--audio` "
            "directory except for the extension. For example, if the transcript file is "
            "`example.srt`, there must be an audio file like `example.wav` under `--audio` "
            "directory.",
        ),
    )
    parser.add_argument(
        "--data",
        type=str,
        help=(
            "Path to a text file containing audio filenames and transcriptions. This option is "
            "used only when `--without-timestamps` is set. Each line must be in the format of "
            "`<audio_path>\t<transcription>`."
        ),
    )
    parser.add_argument("--output", type=str, default="data.json", help="Path to output json file")
    parser.add_argument("--dump", type=str, default="dump", help="Directory to dump audio files")
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        choices=sorted(LANGUAGES.keys()) + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]),
        help="Language of the data",
    )
    parser.add_argument(
        "--timestamp-resolution",
        type=int,
        default=20,
        help=(
            "Timestamp resolution in milliseconds. Defaults to 20ms. Since the native time "
            "resolution of Whisper tokens is 20ms, this option needs to be set to multiples of "
            "20ms."
        ),
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=223,
        help=(
            "Maximum length of prompt in Whisper tokens. Defaults to 223, which equals to "
            "`model.dims.n_text_ctx (=448) // 2 - 1` (-1 is for the special token `sot_prev` and "
            "the other half is for the transcribed tokens)."
        ),
    )
    parser.add_argument(
        "--max-tokens-length",
        type=int,
        default=219,
        help=(
            "Maximum length of text and timestamps tokens. Utterances longer than this will be "
            "skipped. Defaults to 219, which equals to `model.dims.n_text_ctx (=448) // 2 - 5` "
            "(5 is the maximum number of special tokens used other than the `sot_prev`."
        ),
    )
    parser.add_argument(
        "--subsampling-factor-for-silence",
        type=int,
        default=1,
        help=(
            "Subsampling factor for silence. This option is used to reduce the number of silence "
            "utterances. The original Whisper paper uses 1/10 of the number of silence utterances. "
            "Defaults to 1, which means no subsampling."
        ),
    )
    parser.add_argument(
        "--tokenizer-type",
        type=str,
        default="multilingual",
        choices=["multilingual", "english"],
        help=(
            "Type of Whisper tokenizer to use. Tokenizer is used to count the number of tokens "
            "in the transcriptions."
        ),
    )
    parser.add_argument("--normalize-unicode", action="store_true", help="Normalize unicode")
    return parser


def verify_args(args: argparse.Namespace):
    if args.with_timestamps:
        if not args.audio or not args.transcript:
            raise ValueError("--audio and --transcript must be set when --with-timestamps")

        if args.timestamp_resolution % 20 != 0:
            raise ValueError(
                f"Timestamp resolution must be multiples of 20ms. Got {args.timestamp_resolution}"
            )
    else:
        if not args.data:
            raise ValueError("--data must be set when --without-timestamps")


@dataclass
class Utterance:
    """
    A single segment of audio with a transcription. Corresponds to a single chunk in a .srt file.
    """

    audio_path: str
    text: str
    start: Optional[int] = None  # in milliseconds
    end: Optional[int] = None  # in milliseconds


def str_to_milliseconds(s: str) -> int:
    """
    Convert a string in the format of "00:00:00,000" to milliseconds.
    """
    time, miliseconds = s.split(",")
    hours, minutes, seconds = time.split(":")
    hours = int(hours)
    minutes = int(minutes)
    seconds = int(seconds)
    miliseconds = int(miliseconds)
    return (hours * 3600 + minutes * 60 + seconds) * 1000 + miliseconds


def read_utterances_from_srt(
    transcript_path: Union[str, Path], audio_path: Union[str, Path], normalize_unicode: bool = False
) -> List[Utterance]:
    utterances = []
    with open(transcript_path) as f:
        lines = f.readlines()
        timestamps_indices = [i for i, line in enumerate(lines) if " --> " in line]
        timestamps_indices.append(len(lines) + 1)  # a dummy index to make the loop below simpler

        for i in range(len(timestamps_indices) - 1):
            utterance_start = timestamps_indices[i]
            utterance_end = timestamps_indices[i + 1]

            start_time, end_time = lines[utterance_start].strip().split(" --> ")
            start_time = str_to_milliseconds(start_time)
            end_time = str_to_milliseconds(end_time)

            # `utterance_end - 1` corresponds to an index number of the utterance and
            # `utterance_end - 2` corresponds to a newline character, thus the text is included
            # between [`utterance_start + 1`, `utterance_end - 2`).
            text = " ".join(
                [line.strip() for line in lines[utterance_start + 1 : utterance_end - 2]]
            ).strip()
            if normalize_unicode:
                text = unicodedata.normalize("NFKC", text)
            if text == "":
                # With time-aligned data, empty utterances will be created from timestamps later
                # and are not necessary in the first place
                continue

            utterances.append(
                Utterance(audio_path=audio_path, text=text, start=start_time, end=end_time)
            )

    return utterances


def load_utterances_from_audio_and_transcription_dirs(
    audio_dir: str, transcript_dir: str, normalize_unicode: bool = False
) -> List[Utterance]:
    utterances = []
    for audio_path in Path(audio_dir).iterdir():
        speech_id = Path(audio_path).stem
        transcript_path = Path(transcript_dir) / f"{speech_id}.srt"
        if not transcript_path.exists():
            raise FileNotFoundError(f"Transcript file not found: {transcript_path}")

        utterances_for_speech = read_utterances_from_srt(
            transcript_path, audio_path, normalize_unicode
        )
        utterances.extend(utterances_for_speech)
    return utterances


def load_utterances_from_text(text_file: str, normalize_unicode: bool = False) -> List[Utterance]:
    utterances = []
    with open(text_file) as f:
        for line in f:
            audio_path, text = line.strip().split("\t")
            if normalize_unicode:
                text = unicodedata.normalize("NFKC", text)
            utterances.append(Utterance(audio_path=audio_path, text=text))
    return utterances


@dataclass
class Record:
    """
    A single training instance for Whisper.
    `text` can include timestamps in the format of <|0.00|>.
    """

    audio: str
    text: str
    language: str = "en"
    prompt: str = ""


def read_records(path: Union[str, Path]) -> List[Record]:
    records = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            record = Record(
                audio=data["audio"],
                text=data["text"],
                language=data["language"],
                prompt=data["prompt"],
            )
            records.append(record)
    return records


def write_records(records: List[Record], output: Union[str, Path]):
    with open(output, "w") as f:
        for record in records:
            data = {
                "audio": record.audio,
                "text": record.text,
                "language": record.language,
                "prompt": record.prompt,
            }
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


DURATION = 30000  # 30 seconds in milliseconds
SAMPLE_RATE = 16000
DURATION_IN_SAMPLES = int(DURATION * SAMPLE_RATE / 1000)


def get_time_token(time: int, segment_start: int, timestamp_resolution: int = 20) -> str:
    """
    Get the time token for the given time.

    Args:
        time: Time in milliseconds
        segment_start: Start time of the segment in milliseconds
        timestamp_resolution: Timestamp resolution in milliseconds. Defaults to 20ms.

    Returns:
        Time token (e.g. get_time_token(1200, 1000) -> "<|0.20|>")
    """
    if time < segment_start or segment_start + DURATION < time:
        raise ValueError(
            f"Time {time} is out of the segment ({segment_start} - {segment_start + DURATION})"
        )

    time_in_segment = time - segment_start
    nearest_timestamp = (
        round(time_in_segment / timestamp_resolution) * timestamp_resolution
    )  # in milliseconds
    time_token = f"<|{nearest_timestamp / 1000:.2f}|>"
    return time_token


@dataclass
class PromptBufferNode:
    text: str
    num_tokens: int


def get_prompt(prompt_buffer: Deque[PromptBufferNode], max_prompt_length: int) -> str:
    prompt_length = 0
    prompt_buffer_idx = len(prompt_buffer)
    while prompt_buffer_idx >= 1 and prompt_length < max_prompt_length:
        prompt_buffer_idx -= 1
        prompt_length += prompt_buffer[prompt_buffer_idx].num_tokens

    for _ in range(prompt_buffer_idx):
        prompt_buffer.popleft()

    return " ".join([node.text for node in prompt_buffer])


def create_records_with_timestamps_for_audio(
    utterances: List[Utterance],
    dump_dir: str,
    language: str,
    tokenizer: Tokenizer,
    max_prompt_length: int = 223,
    max_tokens_length: int = 219,
    timestamp_resolution: int = 20,
) -> List[Record]:
    Path(dump_dir).mkdir(parents=True, exist_ok=True)
    audio_path = utterances[0].audio_path
    audio = torch.tensor(load_audio(audio_path))
    records = []
    segment_start, segment_end = 0, DURATION  # in milliseconds
    prompt_buffer: Deque[PromptBufferNode] = deque()

    idx = 0
    while idx < len(utterances):
        # If the utterance is included in the segment and longer than the segment, skip it.
        if (
            utterances[idx].start < segment_end
            and utterances[idx].start + DURATION < utterances[idx].end
        ):
            segment_start = utterances[idx].end
            segment_end = segment_start + DURATION
            idx += 1
            continue

        audio_start_idx = int(segment_start * SAMPLE_RATE / 1000)
        segment_audio_path = str((Path(dump_dir) / f"{segment_start}.wav").absolute())
        segment_audio = audio[
            audio_start_idx : min(audio_start_idx + DURATION_IN_SAMPLES, audio.size(0))
        ]
        torchaudio.save(segment_audio_path, segment_audio.unsqueeze(0), SAMPLE_RATE)

        prompt = get_prompt(prompt_buffer, max_prompt_length)

        segment_utterances = []
        while idx < len(utterances) and utterances[idx].start < segment_end:
            segment_utterances.append(utterances[idx])
            idx += 1

        tokens_length = 0
        segment_text = []
        for utterance in segment_utterances:
            start_token = get_time_token(utterance.start, segment_start, timestamp_resolution)
            if utterance.end <= segment_end:
                end_token = get_time_token(utterance.end, segment_start, timestamp_resolution)
                segment_text.extend([start_token, utterance.text, end_token])
                prompt_buffer.append(
                    PromptBufferNode(utterance.text, len(tokenizer.encode(" " + utterance.text)))
                )
                tokens_length += len(tokenizer.encode(utterance.text)) + 2
            else:
                segment_text.append(start_token)
                tokens_length += 1

        if tokens_length > max_tokens_length:
            print(
                f"Skipping {audio_path} ({format_timestamp(segment_start / 1000)}-"
                f"{format_timestamp(segment_end / 1000)}) because it is too long "
                f"({tokens_length} tokens)"
            )
        else:
            record = Record(
                audio=segment_audio_path,
                language=language,
                text="".join(segment_text),
                prompt=prompt,
            )
            records.append(record)

        if len(segment_utterances) == 0:
            segment_start += DURATION
        elif segment_utterances[-1].end <= segment_end:
            segment_start = segment_utterances[-1].end
        else:  # segment_utterances[-1].end > segment_end
            # The text of the last utterance was not included in the segment and will be
            # included in the next segment
            segment_start = segment_utterances[-1].start
            idx -= 1
        segment_end = segment_start + DURATION

    return records


def create_records_with_timestamps(
    utterances: List[Utterance],
    dump_dir: str,
    language: str,
    tokenizer: Tokenizer,
    max_prompt_length: int = 223,
    max_tokens_length: int = 219,
    timestamp_resolution: int = 20,
) -> List[Record]:
    utterances_grouped_by_audio = defaultdict(list)
    for utterance in utterances:
        utterances_grouped_by_audio[utterance.audio_path].append(utterance)

    all_records = []
    for audio_path, utterances_for_audio in utterances_grouped_by_audio.items():
        records = create_records_with_timestamps_for_audio(
            utterances=utterances_for_audio,
            dump_dir=f"{dump_dir}/{Path(audio_path).stem}",
            language=language,
            tokenizer=tokenizer,
            max_prompt_length=max_prompt_length,
            max_tokens_length=max_tokens_length,
            timestamp_resolution=timestamp_resolution,
        )
        all_records.extend(records)

    return all_records


def create_records_without_timestamps(
    utterances: List[Utterance], language: str, tokenizer: Tokenizer, max_tokens_length: int
) -> List[Record]:
    records = []
    for utterance in utterances:
        tokens = tokenizer.encode(utterance.text)
        if len(tokens) > max_tokens_length:
            print(f"Skipping {utterance} because it is too long ({len(tokens)} tokens)")
            continue

        records.append(Record(utterance.audio_path, text=utterance.text, language=language))

    return records


def subsample_silence(records: List[Record], subsampling_factor: int) -> List[Record]:
    if subsampling_factor == 1:
        return records

    silence_records = filter(lambda record: record.text == "", records)
    non_silence_records = filter(lambda record: record.text != "", records)
    return list(non_silence_records) + list(silence_records)[::subsampling_factor]


def main():
    args = get_parser().parse_args()
    verify_args(args)

    tokenizer = get_tokenizer(multilingual=(args.tokenizer_type == "multilingual"))
    Path(args.dump).mkdir(parents=True, exist_ok=True)

    if args.with_timestamps:
        utterances = load_utterances_from_audio_and_transcription_dirs(
            args.audio,
            args.transcript,
            args.normalize_unicode,
        )
        records = create_records_with_timestamps(
            utterances=utterances,
            dump_dir=args.dump,
            language=args.language,
            tokenizer=tokenizer,
            max_prompt_length=args.max_prompt_length,
            max_tokens_length=args.max_tokens_length,
            timestamp_resolution=args.timestamp_resolution,
        )
    else:
        utterances = load_utterances_from_text(args.data, args.normalize_unicode)
        records = create_records_without_timestamps(
            utterances=utterances,
            language=args.language,
            tokenizer=tokenizer,
            max_tokens_length=args.max_tokens_length,
        )

    records = subsample_silence(records, args.subsampling_factor_for_silence)
    write_records(records, args.output)


if __name__ == "__main__":
    main()
