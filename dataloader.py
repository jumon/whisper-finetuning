import re
from typing import List, Optional, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from whisper.audio import CHUNK_LENGTH, N_FRAMES, log_mel_spectrogram, pad_or_trim
from whisper.tokenizer import Tokenizer

from create_data import DataProcessor, Record


class AudioDataset(Dataset):
    def __init__(
        self,
        records: List[Record],
        tokenizer: Tokenizer,
        fp16: bool = True,
        no_timestamps_training: bool = False,
        max_prompt_length: int = 223,  # The maximum number of tokens to use for the prompt
        prompt_use_rate: float = 0.5,
        no_timestamps_rate: float = 0.5,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.fp16 = fp16
        self.no_timestamps_training = no_timestamps_training
        self.max_prompt_length = max_prompt_length
        self.prompt_use_rate = prompt_use_rate
        self.no_timestamps_rate = no_timestamps_rate

        self.num_frames_per_second = N_FRAMES / CHUNK_LENGTH
        # timestamps tokens are from <|0.00|> to <|30.00|> with a step of 0.02
        self.timestamp_pattern = re.compile(r"(<\|[123]?[0-9]\.[0-9][0-9]\|>)")
        self.model_n_text_ctx = 448

    def __len__(self) -> int:
        return len(self.records)

    def _get_prompt_tokens(self, prompt: str) -> List[int]:
        if len(prompt) > 0 and torch.rand(1) < self.prompt_use_rate:
            prompt_tokens = self._encode_text_with_timestamps(prompt)[-self.max_prompt_length :]
            prompt_tokens = [self.tokenizer.sot_prev] + prompt_tokens
        else:
            prompt_tokens = []

        return prompt_tokens

    def _get_special_tokens(
        self, is_text_empty: bool, language: str, no_timestamps: bool
    ) -> List[int]:
        if is_text_empty:
            special_tokens = [self.tokenizer.sot, self.tokenizer.no_speech]
        else:
            special_tokens = [
                self.tokenizer.sot,
                self.tokenizer.special_tokens[f"<|{language}|>"],
                self.tokenizer.special_tokens["<|transcribe|>"],
            ]
            if no_timestamps:
                special_tokens.append(self.tokenizer.no_timestamps)

        return special_tokens

    def _encode_text_with_timestamps(self, text: str) -> List[int]:
        parts = self.timestamp_pattern.split(text)
        parts = [token for token in parts if token != ""]
        tokens = []
        for part in parts:
            if self.timestamp_pattern.fullmatch(part) is not None:
                timestamp = float(part[2:-2])

                # timestamp must be in the range [0, 30] and be a multiple of 0.02 seconds
                if timestamp < 0 or timestamp > 30 or round(timestamp * 100) % 2 != 0:
                    raise ValueError(f"Invalid timestamp: {timestamp}")

                token = self.tokenizer.timestamp_begin + round(timestamp * 100) // 2
                tokens.append(token)
            else:
                tokens.extend(self.tokenizer.encode(part))

        return tokens

    def _get_partial_segment_start(self, tokens: List[int]) -> Optional[float]:
        if (
            len(tokens) >= 2
            and tokens[-2] >= self.tokenizer.timestamp_begin
            and tokens[-1] >= self.tokenizer.timestamp_begin
        ):  # if the last token is a start time token
            return (tokens[-1] - self.tokenizer.timestamp_begin) * 0.02
        else:
            return None

    def _get_text_tokens(self, text: str, no_timestamps: bool) -> Tuple[List[int], Optional[float]]:
        text_tokens = self._encode_text_with_timestamps(text)
        next_partial_segment_start = self._get_partial_segment_start(text_tokens)
        if no_timestamps:
            text_tokens = list(filter(lambda x: x < self.tokenizer.timestamp_begin, text_tokens))

        return text_tokens, next_partial_segment_start

    def _calculate_mel(
        self, audio_path: str, next_partial_segment_start: Optional[float], no_timestamps: bool
    ) -> torch.Tensor:
        mel = log_mel_spectrogram(audio_path)
        if no_timestamps and next_partial_segment_start is not None:
            mel = mel[:, : int(next_partial_segment_start * self.num_frames_per_second)]
        mel = pad_or_trim(mel, N_FRAMES)
        if self.fp16:
            mel = mel.half()

        return mel

    def _construct_decoder_output(
        self, prompt_tokens: List[int], special_tokens: List[int], text_tokens: List[int]
    ) -> List[int]:
        if len(prompt_tokens) == 0:
            decoder_output = special_tokens[1:] + text_tokens + [self.tokenizer.eot]
        else:
            decoder_output = (
                # Mask out the training loss for predicting the prompt tokens. We use "-100" as the
                # default value for the `ignore_index` parameter in
                # `torch.nn.functional.cross_entropy()`. However, we do not mask out the loss for
                # predicting the sot token because our experiment indicates that the original
                # Whisper model assigns a high probability to the sot token after prompt tokens.
                [-100] * (len(prompt_tokens) - 1)
                + special_tokens
                + text_tokens
                + [self.tokenizer.eot]
            )
        return decoder_output

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        record = self.records[index]
        no_timestamps = self.no_timestamps_training or torch.rand(1) < self.no_timestamps_rate

        prompt_tokens = self._get_prompt_tokens(record.prompt)
        text_tokens, next_partial_segment_start = self._get_text_tokens(record.text, no_timestamps)
        is_text_empty = len(text_tokens) == 0
        special_tokens = self._get_special_tokens(is_text_empty, record.language, no_timestamps)

        decoder_input = prompt_tokens + special_tokens + text_tokens
        if len(decoder_input) > self.model_n_text_ctx:
            raise ValueError(f"Input is too long: {record} (length: {len(decoder_input)})")

        decoder_output = self._construct_decoder_output(prompt_tokens, special_tokens, text_tokens)

        mel = self._calculate_mel(record.audio_path, next_partial_segment_start, no_timestamps)

        return (
            mel,
            torch.tensor(decoder_input, dtype=torch.long),
            torch.tensor(decoder_output, dtype=torch.long),
        )


def collate_fn(data):
    x, y_in, y_out = zip(*data)
    x = pad_sequence(x, batch_first=True, padding_value=0)
    y_in = pad_sequence(y_in, batch_first=True, padding_value=0)
    y_out = pad_sequence(y_out, batch_first=True, padding_value=-100)
    return x, y_in, y_out


def get_dataloader(
    json: str,
    tokenizer: Tokenizer,
    batch_size: int = 1,
    fp16: bool = True,
    no_timestamps_training: bool = False,
    max_prompt_length: int = 223,
    prompt_use_rate: float = 0.5,
    no_timestamps_rate: float = 0.5,
    shuffle: bool = True,
) -> DataLoader:
    records = DataProcessor.read_records(json)
    dataset = AudioDataset(
        records,
        tokenizer,
        fp16=fp16,
        no_timestamps_training=no_timestamps_training,
        max_prompt_length=max_prompt_length,
        prompt_use_rate=prompt_use_rate,
        no_timestamps_rate=no_timestamps_rate,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )
