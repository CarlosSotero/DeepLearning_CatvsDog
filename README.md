# 🐶🐱 Cats vs Dogs Classification with CNN and Transfer Learning

Este projeto tem como objetivo construir e comparar modelos de **classificação binária de imagens (Gatos vs Cães)** utilizando **Redes Neurais Convolucionais (CNN)** com TensorFlow/Keras.

O trabalho evolui de um modelo CNN básico para um modelo mais robusto utilizando **Transfer Learning**, demonstrando ganhos significativos de desempenho e generalização.

---

## 📌 Objetivos do Projeto

- Construir um classificador de imagens Gato vs Cachorro
- Aplicar conceitos fundamentais de Visão Computacional
- Comparar um modelo CNN treinado do zero com um modelo baseado em Transfer Learning
- Analisar métricas como Accuracy, AUC e Loss
- Avaliar comportamento de overfitting e generalização

---

## 🗂️ Estrutura do Projeto

```text
.
├── notebooks/
│   ├── Prática_com_Deep_Learning_para_Classificação_de_Imagens.ipynb
│   └── Prática_com_Deep_Learning_para_Classificação_de_Imagens_(versão_melhorada).ipynb
│
├── saved_models/
│   ├── cat_vs_dog.hdf5
│   └── cat_vs_dog_v2.hdf5.hdf5
│
├── README.md
└── requirements.txt
````
---
## 📊 Dataset

- **Dataset:** Cats vs Dogs
- **Classes:** Balanceadas

**Divisão aproximada:**
- Treino: ~10.000 imagens por classe
- Validação: ~2.500 imagens por classe

Imagens corrompidas foram identificadas e removidas antes do treinamento.

---

## 🧠 Modelos Desenvolvidos

### 🔹 Modelo 1 — Prática_com_Deep_Learning_para_Classificação_de_Imagens

- CNN construída do zero  
- Camadas convolucionais + pooling  
- Camadas densas para classificação  
- Sem uso de pesos pré-treinados  

**Resultados aproximados:**
- Validation Accuracy: ~75% – 88%
- Validation AUC: ~0.83 – 0.95

Modelo funcional, porém com limitações de generalização.

---

### 🔹 Modelo 2 — Prática_com_Deep_Learning_para_Classificação_de_Imagens_(versão_melhorada)

- Backbone pré-treinado: **MobileNet**
- `include_top=False`
- Camadas convolucionais congeladas
- `GlobalAveragePooling2D`
- Data Augmentation aplicado durante o treino

**Resultados finais:**
- Validation Accuracy: ~96% – 97%
- Validation AUC: ~0.99+
- Validation Loss: ~0.08 – 0.11

Modelo altamente estável e com excelente capacidade de generalização.

---

## 📈 Métricas e Avaliação

Durante o treinamento foram monitoradas:

- Accuracy (treino e validação)
- AUC (treino e validação)
- Loss (treino e validação)

Os gráficos indicam:
- Curvas de treino e validação próximas
- Ausência de overfitting severo
- Alta estabilidade ao longo das épocas

---

## 💾 Salvamento do Modelo

Os modelos treinados foram salvos para reutilização posterior:

```python
model.save('saved_models/transfer_learning_model.hdf5')
```

## Carregando o Modelo
```python
from keras.models import load_model
model = load_model('saved_models/transfer_learning_model.hdf5', compile=False)
````
## Tecnologias Utilizadas
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Google Colab (GPU quando disponível)

