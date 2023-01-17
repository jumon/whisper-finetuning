import argparse
import json
import unicodedata
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List, Optional, Union

import torch
import torchaudio
from whisper.audio import load_audio
from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE, get_tokenizer
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
            "fine-tuning a Whisper model with time-aligned data. Defaults to True."
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
        "--audio-dir",
        type=str,
        help=(
            "Path to directory containing audio files. This option is used only when "
            "`--with-timestamps` is set. Audio formats that can be read by ffmpeg are supported."
        ),
    )
    parser.add_argument(
        "--transcript-dir",
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
        "--data-file",
        type=str,
        help=(
            "Path to a text file containing audio filenames and transcriptions. This option is "
            "used only when `--without-timestamps` is set. Each line must be in the format of "
            "`<audio_path>\t<transcription>`."
        ),
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        choices=sorted(LANGUAGES.keys()) + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]),
        help="Language of the data",
    )
    parser.add_argument("--output", type=str, default="data.json", help="Path to output json file")
    parser.add_argument(
        "--dump-dir", type=str, default="dump", help="Directory to dump audio files"
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


DURATION = 30000  # 30 seconds in milliseconds
SAMPLE_RATE = 16000
DURATION_IN_SAMPLES = int(DURATION * SAMPLE_RATE / 1000)


@dataclass
class Utterance:
    """
    Representing a single segment of audio with a transcription. Corresponds to a single chunk in a
    .srt file.
    """

    text: str
    start: Optional[int] = None  # in milliseconds
    end: Optional[int] = None  # in milliseconds


@dataclass
class Record:
    """
    A single training instance for Whisper.
    `text` can include timestamps in the format of <|0.00|>.
    """

    audio_path: str
    text: str
    language: str = "en"
    prompt: str = ""


@dataclass
class PromptBufferNode:
    text: str
    num_tokens: int


class DataProcessor:
    def __init__(
        self,
        with_timestamps: bool = True,
        audio_dir: str = None,
        transcript_dir: str = None,
        data_file: str = None,
        language: str = "en",
        output: str = "data.json",
        dump_dir: str = "dump",
        timestamp_resolution: int = 20,
        max_prompt_length: int = 223,
        max_tokens_length: int = 219,
        subsampling_factor_for_silence: int = 1,
        tokenizer_type: str = "multilingual",
        normalize_unicode: bool = False,
    ) -> None:
        self.with_timestamps = with_timestamps
        self.audio_dir = audio_dir
        self.transcript_dir = transcript_dir
        self.data_file = data_file
        self.language = language
        self.output = output
        self.dump_dir = dump_dir
        self.timestamp_resolution = timestamp_resolution
        self.max_prompt_length = max_prompt_length
        self.max_tokens_length = max_tokens_length
        self.subsampling_factor_for_silence = subsampling_factor_for_silence
        self.tokenizer_type = tokenizer_type
        self.normalize_unicode = normalize_unicode

        self._verify_args()

        self.tokenizer = get_tokenizer(multilingual=(self.tokenizer_type == "multilingual"))
        Path(self.dump_dir).mkdir(parents=True, exist_ok=True)

    def _verify_args(self) -> None:
        if self.with_timestamps:
            if not self.audio_dir or not self.transcript_dir:
                raise ValueError(
                    "`audio_dir` and `transcript_dir` must be set when `with_timestamps` is True"
                )

            if self.timestamp_resolution % 20 != 0:
                raise ValueError(
                    "`timestamps_resolution` must be multiples of 20ms. "
                    f"Got {self.timestamp_resolution}"
                )
        else:
            if not self.data_file:
                raise ValueError("`data_file` must be set when `with_timestamps` is False")

        if self.language not in LANGUAGES:
            if self.language in TO_LANGUAGE_CODE:
                self.language = TO_LANGUAGE_CODE[self.language]
            else:
                raise ValueError(f"Unsupported language: {self.language}")

        if self.tokenizer_type not in ["multilingual", "english"]:
            raise ValueError(f"Unsupported tokenizer type: {self.tokenizer_type}")

        if Path(self.output).exists():
            raise ValueError(f"Output file {self.output} already exists")

    def run(self) -> None:
        if self.with_timestamps:
            self._process_with_timestamps()
        else:
            self._process_without_timestamps()

        if self.subsampling_factor_for_silence > 1:
            self._subsample_silence()

    def _process_without_timestamps(self) -> None:
        records = []
        with open(self.data_file) as f:
            for line in f:
                audio_path, text = line.strip().split("\t")
                if self.normalize_unicode:
                    text = unicodedata.normalize("NFKC", text)

                tokens = self.tokenizer.encode(text)
                if len(tokens) > self.max_tokens_length:
                    print(
                        f"Skipping {audio_path} ({text}) because it is too long "
                        f"({len(tokens)} tokens)"
                    )
                    continue

                record = Record(audio_path=audio_path, text=text, language=self.language)
                records.append(record)

        self.write_records(records, self.output)

    def _process_with_timestamps(self) -> None:
        for audio_path in Path(self.audio_dir).iterdir():
            speech_id = Path(audio_path).stem
            transcript_path = Path(self.transcript_dir) / f"{speech_id}.srt"
            if not transcript_path.exists():
                raise FileNotFoundError(f"Transcript file not found: {transcript_path}")

            utterances_for_speech = self.read_utterances_from_srt(
                transcript_path, self.normalize_unicode
            )
            records = self._create_records_with_timestamps(utterances_for_speech, audio_path)
            self.write_records(records, self.output)

    @staticmethod
    def read_utterances_from_srt(
        transcript_path: Union[str, Path], normalize_unicode: bool = False
    ) -> List[Utterance]:
        utterances = []
        with open(transcript_path) as f:
            lines = f.readlines()
            timestamps_indices = [i for i, line in enumerate(lines) if " --> " in line]
            timestamps_indices.append(len(lines) + 1)  # a dummy index to make the loop below simple

            for i in range(len(timestamps_indices) - 1):
                utterance_start = timestamps_indices[i]
                utterance_end = timestamps_indices[i + 1]

                start_time, end_time = lines[utterance_start].strip().split(" --> ")
                start_time = DataProcessor.str_to_milliseconds(start_time)
                end_time = DataProcessor.str_to_milliseconds(end_time)

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

                utterances.append(Utterance(text=text, start=start_time, end=end_time))

        return utterances

    def _create_records_with_timestamps(
        self, utterances: List[Utterance], audio_path: Path
    ) -> List[Record]:
        audio = torch.tensor(load_audio(audio_path))
        dump_dir = Path(self.dump_dir) / audio_path.stem
        dump_dir.mkdir(parents=True, exist_ok=True)
        records = []
        prompt_buffer: Deque[PromptBufferNode] = deque()
        segment_start, segment_end = 0, DURATION  # in milliseconds

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

            segment_audio_path = self._save_segment_audio(audio, segment_start, dump_dir)
            prompt = self._get_prompt(prompt_buffer)

            segment_utterances = []
            while idx < len(utterances) and utterances[idx].start < segment_end:
                segment_utterances.append(utterances[idx])
                idx += 1

            tokens_length = 0
            segment_text = []
            for utterance in segment_utterances:
                start_token = self._get_time_token(utterance.start, segment_start)
                if utterance.end <= segment_end:
                    end_token = self._get_time_token(utterance.end, segment_start)
                    segment_text.extend([start_token, utterance.text, end_token])
                    prompt_buffer.append(
                        PromptBufferNode(
                            utterance.text, len(self.tokenizer.encode(" " + utterance.text))
                        )
                    )
                    tokens_length += len(self.tokenizer.encode(utterance.text)) + 2
                else:
                    segment_text.append(start_token)
                    tokens_length += 1

            if tokens_length > self.max_tokens_length:
                print(
                    f"Skipping {audio_path} ({format_timestamp(segment_start / 1000)}-"
                    f"{format_timestamp(segment_end / 1000)}) because it is too long "
                    f"({tokens_length} tokens)"
                )
            else:
                record = Record(
                    audio_path=segment_audio_path,
                    language=self.language,
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

    def _save_segment_audio(self, audio: torch.Tensor, segment_start: int, dump_dir: Path) -> str:
        audio_start_idx = int(segment_start * SAMPLE_RATE / 1000)
        segment_audio_path = str((dump_dir / f"{segment_start}.wav").absolute())
        segment_audio = audio[
            audio_start_idx : min(audio_start_idx + DURATION_IN_SAMPLES, audio.size(0))
        ]
        torchaudio.save(segment_audio_path, segment_audio.unsqueeze(0), SAMPLE_RATE)
        return segment_audio_path

    @staticmethod
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

    def _get_time_token(self, time: int, segment_start: int) -> str:
        """
        Get the time token for the given time.

        Args:
            time: Time in milliseconds
            segment_start: Start time of the segment in milliseconds

        Returns:
            Time token (e.g. self._get_time_token(1200, 1000) -> "<|0.20|>")
        """
        if time < segment_start or segment_start + DURATION < time:
            raise ValueError(
                f"Time {time} is out of the segment ({segment_start} - {segment_start + DURATION})"
            )

        time_in_segment = time - segment_start
        nearest_timestamp = (
            round(time_in_segment / self.timestamp_resolution) * self.timestamp_resolution
        )  # in milliseconds
        time_token = f"<|{nearest_timestamp / 1000:.2f}|>"
        return time_token

    def _get_prompt(self, prompt_buffer: Deque[PromptBufferNode]) -> str:
        prompt_length = 0
        prompt_buffer_idx = len(prompt_buffer)
        while prompt_buffer_idx >= 1 and prompt_length < self.max_prompt_length:
            prompt_buffer_idx -= 1
            prompt_length += prompt_buffer[prompt_buffer_idx].num_tokens

        for _ in range(prompt_buffer_idx):
            prompt_buffer.popleft()

        return " ".join([node.text for node in prompt_buffer])

    @staticmethod
    def read_records(path: Union[str, Path]) -> List[Record]:
        records = []
        with open(path) as f:
            for line in f:
                data = json.loads(line)
                record = Record(
                    audio_path=data["audio_path"],
                    text=data["text"],
                    language=data["language"],
                    prompt=data["prompt"],
                )
                records.append(record)
        return records

    @staticmethod
    def write_records(records: List[Record], path: Union[str, Path]) -> None:
        with open(path, "a") as f:
            for record in records:
                data = {
                    "audio_path": record.audio_path,
                    "text": record.text,
                    "language": record.language,
                    "prompt": record.prompt,
                }
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def _subsample_silence(self) -> None:
        records = self.read_records(self.output)

        silence_records = filter(lambda record: record.text == "", records)
        non_silence_records = filter(lambda record: record.text != "", records)
        filtered_records = (
            list(non_silence_records)
            + list(silence_records)[:: self.subsampling_factor_for_silence]
        )

        Path(self.output).unlink()
        self.write_records(filtered_records, self.output)


def main():
    args = get_parser().parse_args()
    processor = DataProcessor(
        with_timestamps=args.with_timestamps,
        audio_dir=args.audio_dir,
        transcript_dir=args.transcript_dir,
        data_file=args.data_file,
        language=args.language,
        output=args.output,
        dump_dir=args.dump_dir,
        timestamp_resolution=args.timestamp_resolution,
        max_prompt_length=args.max_prompt_length,
        max_tokens_length=args.max_tokens_length,
        subsampling_factor_for_silence=args.subsampling_factor_for_silence,
        tokenizer_type=args.tokenizer_type,
        normalize_unicode=args.normalize_unicode,
    )
    processor.run()


if __name__ == "__main__":
    main()
