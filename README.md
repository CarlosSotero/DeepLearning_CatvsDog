# 🐱🐶 Classificação de Imagens: Cats vs Dogs com CNN (Keras)

Este projeto implementa um modelo de **Deep Learning** para **classificação binária de imagens (gatos vs cães)** utilizando **Redes Neurais Convolucionais (CNN)** com **TensorFlow/Keras**. O objetivo principal é explorar o pipeline completo de visão computacional: preparação dos dados, treinamento, avaliação e salvamento do modelo.

O trabalho foi desenvolvido em ambiente **Google Colab**, considerando limitações reais de GPU e boas práticas de treinamento.

---

## 📌 Objetivo do Projeto

Construir e treinar um modelo capaz de identificar corretamente se uma imagem contém um **gato** ou um **cão**, avaliando o desempenho com métricas apropriadas para classificação binária, como **AUC (Area Under the Curve)**.

---

## 🧠 Abordagem Utilizada

* Rede Neural Convolucional (CNN)
* Framework: **TensorFlow / Keras**
* Função de perda: `binary_crossentropy`
* Otimizador: `RMSprop`
* Métricas: `accuracy` e `AUC`
* Pooling global com `GlobalAveragePooling2D`
* Treinamento via **ImageDataGenerator**

---

## 🗂️ Dataset

O conjunto de dados é composto por imagens de gatos e cães, organizadas em pastas de **treino** e **validação**.

Distribuição após limpeza de imagens corrompidas:

* **Treino**

  * Gatos: 9.999
  * Cães: 9.999

* **Validação**

  * Gatos: 2.500
  * Cães: 2.500

---

## 🧹 Pré-processamento

* Redimensionamento das imagens
* Normalização dos pixels
* Carregamento em batches via geradores

---

## 🏋️ Treinamento

* Número de épocas: **15**
* Steps por época: **1000**
* Validação a cada época

### 📊 Resultados finais

* **AUC Treino:** ~0.85
* **AUC Validação:** ~0.89
* Curvas de treino e validação próximas, indicando **boa capacidade de generalização** e ausência de overfitting severo.

---

## 📈 Evolução da AUC

O gráfico abaixo mostra a evolução da métrica AUC ao longo das épocas para treino e validação:

<img width="543" height="429" alt="image" src="https://github.com/user-attachments/assets/03bd2c83-9812-4d5b-b782-2cf6e2d76c07" />

*

O comportamento das curvas indica aprendizado consistente, com validação acompanhando o treino de forma estável.

---

## 💾 Salvamento do Modelo

Para evitar a necessidade de re-treinamento a cada execução, o modelo é salvo após o treinamento:

```python
# Salvando o modelo
model.save('/content/Cats_vs_Dogs/saved_models/cat_vs_dog.hdf5')

# Carregando o modelo salvo
from keras.models import load_model
model = load_model('/content/Cats_vs_Dogs/saved_models/cat_vs_dog.hdf5', compile=False)
```

---

## 🛠️ Tecnologias Utilizadas

* Python
* TensorFlow / Keras
* Google Colab
* Matplotlib
* NumPy

---

## 🚀 Próximos Passos

* Aplicar **Data Augmentation**
* Utilizar **Transfer Learning** (ex: MobileNet, ResNet)
* Ajuste fino de hiperparâmetros
* Avaliação com matriz de confusão

---

## 📚 Aprendizados

Este projeto reforça conceitos fundamentais de visão computacional, como:

* Funcionamento de CNNs
* Importância da AUC em problemas desbalanceados
* Impacto do pré-processamento e da arquitetura no desempenho
* Boas práticas de salvamento e reutilização de modelos

---

📌 *Projeto desenvolvido com foco educacional e prática em Deep Learning aplicado a imagens.*
