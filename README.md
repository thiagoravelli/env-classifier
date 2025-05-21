# LLaMA-2 Environmental Text Classifier (QLoRA)

Fine-tuning de **LLaMA-2-7B** em 4-bit para classificar textos ambientais
em quatro categorias: Emissões, Energia Renovável, Conservação,
Política Climática.

|          | Valor |
|----------|-------|
| Dataset  | EcoVerse (3k tweets) |
| Acurácia | 88 % |
| Tamanho  | 3.8 GB (LoRA+4-bit) |
| GPU      | T4 (15 GB) – 3 épocas, 45 min |

## Reproduzir
```bash
conda env create -f environment.yml
python src/download_data.py
...
