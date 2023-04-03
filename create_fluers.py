import unicodedata
from datasets import load_dataset
import argparse
from scipy.io.wavfile import write
from create_data import DataProcessor, Record
from whisper.tokenizer import get_tokenizer
flerus_languages_mapping = {
    "da" : "da_dk",
    "de" : "de_de",
    "en" : "en_gb",} # add more languages here


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="/work3/s183954/fleurs")
    parser.add_argument("--language", type=str, default="da")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--tokenizer_type", type=str, default="multilingual", choices=["multilingual", "english"])
    parser.add_argument("--normalize_unicode", type=bool,default=False)
    parser.add_argument("--max_tokens_length", type=int, default=219)
    parser.add_argument("--output_audio", type=str, default="/work3/s183954/datasets/fleurs")
    parser.add_argument("--output" ,type=str, default="flerus_train")
    return parser.parse_args()

def main(args):
    args = parse_args()
    dataset = load_dataset("google/fleurs", flerus_languages_mapping[args.language], split=args.split, cache_dir=args.path)
    dataset = dataset.remove_columns(['id', 'num_samples', 'path',
                                                     'transcription', 'gender',
                                                     'lang_id', 'language', 'lang_group_id'])
    
    tokenizer = get_tokenizer(multilingual=(args.tokenizer_type == "multilingual"))
    
    records = []
    for item in range(dataset.shape[0]):
        text = dataset[item]['raw_transcription']
        audio = dataset[item]['audio']['array']
        if args.normalize_unicode:
            text = unicodedata.normalize("NFKC", text)
        tokens = tokenizer.encode(text)
        if len(tokens) > args.max_tokens_length:
            print(
                    f"Skipping {audio} ({text}) because it is too long "
                    f"({len(tokens)} tokens)"
                )
            continue
        # save audio as wav file
        path = f"{args.output_audio}_audio/{args.split}/{item}.wav"
        write(path, 16000, audio)
        record = Record(audio_path=path, text=text, language=args.language)
        records.append(record)
    DataProcessor.write_records(records, args.output)

if __name__ == "__main__":
    args = parse_args()
    main(args=args)