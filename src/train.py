import argparse, torch
from datasets import load_from_disk
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from torch.nn import CrossEntropyLoss

# ---------- métricas ----------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }

# ---------- Trainer ponderado ----------
class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fct = CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float).to(self.args.device)
        )

    # aceita kwargs extras (ex.: num_items_in_batch)
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = self.loss_fct(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss

# ---------- script principal ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--model_name", default="bert-base-uncased")
    parser.add_argument("--class_weights", nargs=3, type=float,
                        default=[1.28, 1.0, 1.86])
    args = parser.parse_args()

    # ---- dataset ----
    data = load_from_disk("data/processed")
    if "label" in data["train"].column_names:
        data = data.rename_column("label", "labels")

    id2label = {0: "risk", 1: "neutral", 2: "opportunity"}

    # ---- modelo base ----
    backbone = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(id2label),
        id2label=id2label,
        label2id={v: k for k, v in id2label.items()},
    )

    # ---- LoRA ----
    lora_cfg = LoraConfig(
        r=args.lora_rank,
        target_modules=["query", "value"],
        lora_alpha=32,
        lora_dropout=0.1,
    )
    model = get_peft_model(backbone, lora_cfg)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # ---- TrainingArguments ----
    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epochs,
        learning_rate=1e-4,
        logging_steps=25,
        label_names=["labels"],
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=2
    )

    trainer = WeightedTrainer(
        class_weights=args.class_weights,
        model=model,
        args=targs,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("Validation metrics:", metrics)

    model.save_pretrained(f"{args.output_dir}/best")
    tokenizer.save_pretrained(f"{args.output_dir}/best")
    print("✓ Modelo BERT + LoRA (loss ponderado) treinado e salvo.")

if __name__ == "__main__":
    main()
