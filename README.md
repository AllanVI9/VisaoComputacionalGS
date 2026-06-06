# 🌍 Land Cover Classification with CNN (PyTorch)

## 👥 Integrantes do Grupo
- Allan Von Ivanov - Rm98705
- Arthur Candido Cabreu - Rm98283
- Giuliano Romaneto - Rm99694
- Bianca Carvalhop Dancs Firsoff - Rm551645

Projeto de **Visão Computacional com Deep Learning** para classificação de imagens de cobertura terrestre em 4 classes:

- 🌾 Agriculture
- 🌲 Forest
- 🏠 Residential
- 🌊 Water

O objetivo é treinar e comparar diferentes arquiteturas de CNN para identificar automaticamente o tipo de uso do solo em imagens diretas do satelite.

## 🚀 Visão Geral

Este projeto implementa um pipeline completo de Machine Learning:

- 📦 Carregamento de dataset com `ImageFolder`
- 🔄 Pré-processamento e normalização de imagens
- ✂️ Split em treino, validação e teste
- 🧠 Treinamento de CNN simples e CNN profunda
- 📊 Avaliação com accuracy e loss
- 📉 Matriz de confusão para análise de erros
- 🖼️ Visualização de previsões do modelo

## 🧪 Comandos
- Para instalar os requerimentos:
  ```python
  pip install -r requirements.txt
- Para rodar o código e treinar os modelos
  ```python
  python run_all.py

## 🧠 Arquitetura dos Modelos

### 🔹 SimpleCNN
Modelo leve com 3 blocos de convolução:

- Conv2D → ReLU → MaxPool
- Fully Connected layers

📌 Foco: baseline rápido e eficiente

### 🔸 DeepCNN
Modelo mais profundo com maior capacidade de extração de features:

- Mais camadas convolucionais
- Melhor capacidade de generalização
- Maior sensibilidade a padrões complexos

📌 Foco: maior performance e precisão

## 📊 Dataset
Estrutura esperada:
```bash
dataset/
├── Agriculture/
├── Forest/
├── Residential/
└── Water/
```


Cada pasta contém imagens correspondentes à classe.


## ⚙️ Tecnologias

- Python 3.10+
- PyTorch
- TorchVision
- Matplotlib
- NumPy
- Scikit-learn
- tqdm


## 🏗️ Pipeline do Projeto

### 1. Preparação do Dataset
- Resize para 64x64
- Normalização (mean=0.5, std=0.5)

### 2. Split
- 70% treino
- 15% validação
- 15% teste


### 3. Treinamento
- Otimizador: Adam
- Loss: CrossEntropyLoss
- Epochs: 10 (configurável)


### 4. Avaliação

O modelo é avaliado com:

- Accuracy geral
- Classification Report
- Matriz de Confusão

## 📈 Resultados

Durante os testes:

- SimpleCNN: ~90% accuracy
- DeepCNN: ~95% accuracy

📌 O DeepCNN apresentou melhor capacidade de generalização em classes mais complexas.

## 📉 Matriz de Confusão

A matriz mostra:

- ✔ Acertos na diagonal principal
- ❌ Erros entre classes específicas

Exemplo de interpretação:

- Forest sendo confundido com Agriculture indica similaridade visual entre padrões vegetais
- Residential pode ser confundido com Agriculture dependendo da textura da imagem

## 🖼️ Visualização de Resultados

O projeto exibe:

- Previsões lado a lado (real vs predito)
- Comparação de loss entre modelos
- Comparação de accuracy por epoch
