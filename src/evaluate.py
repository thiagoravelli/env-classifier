import argparse, pathlib, numpy as np, matplotlib.pyplot as plt
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns, torch


def batch_predict(texts, tokenizer, model, device, batch_size=32):
    """Faz inferência em lotes para economizar memória."""
    preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tokenizer(
                texts[i : i + batch_size],
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits
            preds.extend(logits.argmax(dim=-1).cpu().numpy())
    return np.array(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="Pasta do modelo salvo (e.g. outputs/best)")
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- carrega dataset ----
    ds = load_from_disk("data/processed")
    test_split = ds["test"]
    y_true = np.array(test_split["labels"] if "labels" in test_split.column_names else test_split["label"])

    # ---- carrega modelo + tokenizer ----
    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint).to(device)

    # ---- predições em lote ----
    preds = batch_predict(test_split["text"], tok, model, device, args.batch_size)

    # ---- métricas e matriz de confusão ----
    print(classification_report(y_true, preds, digits=4))
    cm = confusion_matrix(y_true, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predito"); plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig("confusion2.png")
    print("✓ Relatório impresso e confusion.png salvo.")

if __name__ == "__main__":
    main()
