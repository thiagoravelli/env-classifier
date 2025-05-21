"""
Baixa o dataset climatebert/climate_sentiment do Hugging Face,
une os splits train + test e salva em data/raw/climate_sentiment.csv
"""

import argparse
import pathlib
import pandas as pd
from datasets import load_dataset


def download_climate(dest: pathlib.Path) -> None:
    """Carrega o dataset Climate Sentiment e grava em CSV único."""
    # carrega os dois splits públicos
    ds_train = load_dataset("climatebert/climate_sentiment", split="train")
    ds_test  = load_dataset("climatebert/climate_sentiment", split="test")

    # converte para pandas e concatena
    df = pd.concat([ds_train.to_pandas(), ds_test.to_pandas()],
                   ignore_index=True)

    out = dest / "raw" / "climate_sentiment.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"✓ Dataset salvo em {out} ({len(df)} linhas)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="climate",
                    help="Use 'climate' para climate_sentiment")
    args = ap.parse_args()

    root = pathlib.Path(__file__).resolve().parents[1] / "data"

    if args.dataset == "climate":
        download_climate(root)
    else:
        raise ValueError("Dataset não suportado. Use --dataset climate")


if __name__ == "__main__":
    main()
