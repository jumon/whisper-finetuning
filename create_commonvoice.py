import unicodedata
import pandas as pd
import argparse

from create_data import Record, DataProcessor
from whisper.tokenizer import get_tokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="/work3/s212373/common_voice_11")
    parser.add_argument("--language", type=str, default="da")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--tokenizer_type", type=str, default="multilingual", choices=["multilingual", "english"])
    parser.add_argument("--normalize_unicode", type=bool,default=False)
    parser.add_argument("--max_tokens_length", type=int, default=219)
    parser.add_argument("--output", type=str, default="commonvoice_train")
    return parser.parse_args()
    

def process_dataset(dataset : pd.DataFrame, tokenizer, args):
    records = []
    for index, row in dataset.iterrows():
        text = row['sentence']
        audio_path = row['full_path']
        if args.normalize_unicode:
            text = unicodedata.normalize("NFKC", text)
        tokens = tokenizer.encode(text)
        if len(tokens) > args.max_tokens_length:
            print(
                    f"Skipping {audio_path} ({text}) because it is too long "
                    f"({len(tokens)} tokens)"
                )
            continue
        record = Record(audio_path=audio_path, text=text, language=args.language)
        records.append(record)
    print(f"Saving {len(records)} records to {args.output}")
    DataProcessor.write_records(records, args.output)


def main(args):
    dataset = pd.read_csv(f'{args.path}/{args.language}/{args.split}.tsv', sep='\t')
    dataset.drop(['client_id', 'up_votes', 'down_votes', 'age', 'gender', 'locale', 'segment'], axis=1, inplace=True)
    
    dataset['path'].dropna(inplace=True)
    dataset['sentence'].dropna(inplace=True)
    dataset = dataset.reset_index(drop=True)

    dataset['full_path'] = dataset['path'].apply(lambda x: f'{args.path}/{args.language}/clips/{x}')
    tokenizer = get_tokenizer(multilingual=(args.tokenizer_type == "multilingual"))

    process_dataset(dataset, tokenizer, args)


if __name__ == "__main__":
    args = parse_args()
    main(args=args)