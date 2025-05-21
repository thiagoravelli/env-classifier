import argparse, pathlib, os
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from dotenv import load_dotenv

load_dotenv()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--max_length", type=int, default=256)
    args = ap.parse_args()

    data_dir = pathlib.Path(__file__).parents[1] / "data"
    ds = load_dataset("csv", data_files=str(data_dir / "raw/climate_sentiment.csv"))
    ds = ds["train"].train_test_split(test_size=0.2, seed=42)
    ds = DatasetDict({
        "train": ds["train"],
        "test": ds["test"].train_test_split(test_size=0.5, seed=42)["train"],
        "validation": ds["test"].train_test_split(test_size=0.5, seed=42)["test"],
    })

    tok = AutoTokenizer.from_pretrained(args.model_name)
    def tokenize(x):
        return tok(x["text"], truncation=True, padding="max_length",
                   max_length=args.max_length)
    ds_tok = ds.map(tokenize, batched=True)
    ds_tok.save_to_disk(data_dir / "processed")
    print("✓ Dataset tokenizado e salvo.")

if __name__ == "__main__":
    main()
