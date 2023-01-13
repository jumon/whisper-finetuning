import argparse
import copy
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import whisper
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from whisper import Whisper
from whisper.tokenizer import get_tokenizer

from dataloader import get_dataloader


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune a Whisper model for ASR")
    # Dataloader-related arguments
    parser.add_argument(
        "--train-json",
        type=str,
        required=True,
        help="Path to a json file containing training data",
    )
    parser.add_argument(
        "--dev-json",
        type=str,
        required=True,
        help="Path to a json file containing development data",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--dev-batch-size", type=int, default=16, help="Batch size for validation")
    parser.add_argument(
        "--no-timestamps-training",
        action="store_true",
        help="Always use the no-timestamps training mode",
    )
    parser.add_argument(
        "--prompt-use-rate",
        type=float,
        default=0.5,
        help="How often to use prompts for conditioning the generation",
    )
    parser.add_argument(
        "--no-timestamps-rate",
        type=float,
        default=0.5,
        help=(
            "How often to use the no-timestamps mode. Only used if --no-timestamps-training "
            "is NOT set"
        ),
    )

    # Training-related arguments
    parser.add_argument(
        "--save-dir", type=str, default="output", help="directory to save the model"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="device to use for training",
    )
    parser.add_argument(
        "--model",
        default="large",
        choices=whisper.available_models(),
        help="name of the Whisper model to use",
    )
    parser.add_argument("--train-only-decoder", action="store_true", help="train only the decoder")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for training")
    parser.add_argument(
        "--accum-grad-steps",
        type=int,
        default=64,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=500,
        help="Number of warmup steps for learning rate scheduler",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for gradient clipping",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=5000,
        help="Number of training steps",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=500,
        help="Number of steps to evaluate the model",
    )
    parser.add_argument(
        "--save-all-checkpoints",
        action="store_true",
        help="Save all checkpoints instead of only the best one",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser


def train_step(
    model: Whisper,
    train_iter: Iterator,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    accum_grad_steps: int,
    train_only_decoder: bool,
    max_grad_norm: float,
) -> Tuple[float, Iterator]:
    model.train()
    total_loss = 0
    for _ in range(accum_grad_steps):
        x, y_in, y_out = next(train_iter)
        x, y_in, y_out = x.to(model.device), y_in.to(model.device), y_out.to(model.device)

        if train_only_decoder:
            with torch.no_grad():
                audio_features = model.embed_audio(x)
        else:
            audio_features = model.embed_audio(x)
        logits = model.logits(y_in, audio_features=audio_features)
        loss = F.cross_entropy(logits.transpose(1, 2), y_out)

        loss = loss / accum_grad_steps
        loss.backward()
        total_loss += loss.item()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    return total_loss


@torch.no_grad()
def evaluate(model: Whisper, dev_loader: DataLoader) -> float:
    model.eval()
    total_loss = 0
    for x, y_in, y_out in tqdm(dev_loader):
        x, y_in, y_out = x.to(model.device), y_in.to(model.device), y_out.to(model.device)
        logits = model(x, y_in)
        loss = F.cross_entropy(logits.transpose(1, 2), y_out)
        total_loss += loss.item()
    return total_loss / len(dev_loader)


def save_model(model: Whisper, save_path: str) -> None:
    # save model in half precision to save space
    model = copy.deepcopy(model).half()
    # save model weights and config in a dictionary that can be loaded with `whisper.load_model`
    torch.save({"model_state_dict": model.state_dict(), "dims": asdict(model.dims)}, save_path)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def save_args(args: argparse.Namespace, path: str) -> None:
    with open(path, "w") as f:
        f.write(json.dumps(vars(args), indent=4, ensure_ascii=False))


def infinite_iter(data_loader: DataLoader) -> Iterator:
    while True:
        for batch in data_loader:
            yield batch


def main_loop(
    model: Whisper,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    args: argparse.Namespace,
) -> None:
    min_loss = evaluate(model, dev_loader)
    print(f"Initial loss: {min_loss}")
    pbar = tqdm(range(1, args.train_steps + 1))
    train_iter = infinite_iter(train_loader)
    for step in pbar:
        train_loss = train_step(
            model,
            train_iter,
            optimizer,
            scheduler,
            args.accum_grad_steps,
            args.train_only_decoder,
            args.max_grad_norm,
        )
        pbar.set_postfix({"loss": train_loss})

        if step % args.eval_steps == 0:
            eval_loss = evaluate(model, dev_loader)
            tqdm.write(f"Step {step}: validation loss={eval_loss}")
            if eval_loss < min_loss:
                min_loss = eval_loss
                save_model(model, f"{args.save_dir}/best_model.pt")

            if args.save_all_checkpoints:
                save_model(model, f"{args.save_dir}/step{step}.pt")


def main():
    args = get_parser().parse_args()
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    save_args(args, f"{args.save_dir}/args.json")

    tokenizer = get_tokenizer(multilingual=".en" not in args.model, task="transcribe")
    model = whisper.load_model(args.model, args.device)
    #  -1 is for the special token `sot_prev` and the other half is for the transcribed tokens
    max_prompt_length = model.dims.n_text_ctx // 2 - 1

    fp16 = args.device == "cuda"
    train_loader = get_dataloader(
        json=args.train_json,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        fp16=fp16,
        no_timestamps_training=args.no_timestamps_training,
        max_prompt_length=max_prompt_length,
        prompt_use_rate=args.prompt_use_rate,
        no_timestamps_rate=args.no_timestamps_rate,
        shuffle=True,
    )
    dev_loader = get_dataloader(
        json=args.dev_json,
        tokenizer=tokenizer,
        batch_size=args.dev_batch_size,
        fp16=fp16,
        no_timestamps_training=args.no_timestamps_training,
        max_prompt_length=max_prompt_length,
        # always use prompts and timestamps for validation to make it deterministic
        prompt_use_rate=1.0,
        no_timestamps_rate=0.0,
        shuffle=False,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.train_steps
    )

    main_loop(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        args=args,
    )


if __name__ == "__main__":
    main()
