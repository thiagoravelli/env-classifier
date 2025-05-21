import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- Caminho do checkpoint salvo ---
CHECKPOINT = "outputs/best"

# --- Carrega tokenizer e modelo ---
tok = AutoTokenizer.from_pretrained(CHECKPOINT)
model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT)

# --- Seleciona CPU ou GPU automaticamente ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

# --- Mapeamento de rótulos (Risk / Neutral / Opportunity) ---
labels = {0: "Risco", 1: "Neutro", 2: "Oportunidade"}

def classify(text: str) -> str:
    """Classifica o texto em lote único."""
    with torch.no_grad():
        enc = tok(text, return_tensors="pt", truncation=True, padding=True).to(device)
        pred_id = model(**enc).logits.argmax(dim=-1).item()
    return labels.get(pred_id, "Desconhecido")

# --- Interface Gradio ---
demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(lines=3, placeholder="Digite um texto sobre clima..."),
    outputs="text",
    title="Classificador de Sentimento Climático (BERT + LoRA)",
    description="Prediz se o texto expressa Risco, Neutralidade ou Oportunidade em relação às mudanças climáticas."
)

if __name__ == "__main__":
    demo.launch()
