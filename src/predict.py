import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def load_model(chk):
    tok = AutoTokenizer.from_pretrained(chk)
    model = AutoModelForSequenceClassification.from_pretrained(chk,
                                       device_map="auto")
    return tok, model

def classify(txt, tok, model):
    with torch.no_grad():
        out = model(**tok(txt, return_tensors="pt", truncation=True).to("cuda"))
        label_id = out.logits.argmax(-1).item()
    return label_id

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--text", required=True)
    args = ap.parse_args()

    tokenizer, model = load_model(args.checkpoint)
    print(classify(args.text, tokenizer, model))
