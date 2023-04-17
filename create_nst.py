import unicodedata
from datasets import load_dataset
import argparse
from scipy.io.wavfile import write
from create_data import DataProcessor, Record
from whisper.tokenizer import get_tokenizer
import pandas as pd
flerus_languages_mapping = {
    "da" : "da_dk",
    "de" : "de_de",
    "en" : "en_gb",} # add more languages here


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="/work3/s183954/NST_dk/")
    parser.add_argument("--language", type=str, default="da")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--tokenizer_type", type=str, default="multilingual", choices=["multilingual", "english"])
    parser.add_argument("--normalize_unicode", type=bool,default=False)
    parser.add_argument("--max_tokens_length", type=int, default=219)
    parser.add_argument("--output" ,type=str, default="nst_train")
    return parser.parse_args()

def main(args):
    args = parse_args()
    if args.split == "train":
        df = pd.read_csv(args.path + "NST_dk_clean.csv", sep=",", low_memory=False)
        path = args.path + "dk/"
        file_names = df["filename_both_channels"]
    elif args.split == "test":
        df = pd.read_csv(args.path + "supplement_dk_clean.csv", sep=",", low_memory=False)
        path = args.path + "supplement_dk/testdata/audio/"
        file_names = df["filename_channel_1"]
    else:
        raise ValueError("split must be either train or test")
    
    print(f"Processing {args.split} data")
    text_list = df["text"]
    tokenizer = get_tokenizer(multilingual=(args.tokenizer_type == "multilingual"))
    print("processing records")
    records = []
    for item in range(df.shape[0]):
        text = text_list[item]
        filename = file_names[item]
        folder = filename.split("_")[0]

        if args.split != "train":
            filename = filename.split("_")[1]
        
        auido_path = f"{path}{folder}/{filename.lower()}"
        if args.normalize_unicode:
            text = unicodedata.normalize("NFKC", text)
        tokens = tokenizer.encode(text)
        if len(tokens) > args.max_tokens_length:
            print(
                    f"Skipping {path} ({text}) because it is too long "
                    f"({len(tokens)} tokens)"
                )
            continue
        record = Record(audio_path=auido_path, text=text, language=args.language)
        records.append(record)
    print(f"Saving {len(records)} records to {args.output}")
    DataProcessor.write_records(records, args.output)

if __name__ == "__main__":
    args = parse_args()
    main(args=args)