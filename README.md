# Relatório — TP2 SAVI-MNIST  
## Tarefa 1: Classificação | Tarefa 2: Geração + Análise de Dataset “MNIST-Detection”


---

## 1) Enquadramento e objetivo
Este trabalho tem duas partes:

- **Tarefa 1 (Classificação):** treinar e avaliar um modelo CNN para classificar dígitos MNIST (0–9), produzindo métricas e visualizações (matriz de confusão, tabela de resultados, curvas de loss).
- **Tarefa 2 (MNIST-Detection):** gerar um dataset de “deteção” com imagens maiores contendo vários dígitos (com *bounding boxes*) e analisar estatísticas desse dataset (distribuições, histogramas e mosaicos com BBs).

---

## 2) Organização do projeto (ficheiros considerados)

### Tarefa 1 — Classificação
- `main_classification.py` — ponto de entrada: cria dataset/modelo/trainer e executa treino + avaliação.
- `dataset.py` — carrega imagens e labels; devolve tensores para o `DataLoader`.
- `model.py` — define arquiteturas (ex.: `ModelBetterCNN`).
- `trainer.py` — treino, checkpoints, avaliação, matriz de confusão e tabela de métricas.

### Tarefa 2 — MNIST-Detection
- `generate_data.py` — gera imagens “cena” (ex.: 128×128) com dígitos e labels com *bounding boxes*.
- `main_dataset_stats.py` — calcula estatísticas e cria visualizações do dataset gerado.

---

# 3) Tarefa 1 — Classificação MNIST

## 3.1 Pipeline (`main_classification.py`)
O script:
1. Lê argumentos (dataset, nº épocas, batch size, pasta de resultados, etc.).
2. Cria os datasets de treino e teste com `Dataset(..., is_train=True/False)`.
3. Instancia o modelo (por defeito `ModelBetterCNN`).
4. Cria `Trainer(...)`, chama `train()` e depois `evaluate()`.

---

## 3.2 Dataset (`dataset.py`)
### Estrutura esperada
O loader assume esta organização:
- `.../train/images/*.jpg`
- `.../train/labels.txt`
- `.../test/images/*.jpg`
- `.../test/labels.txt`

Em cada linha do `labels.txt`, o código usa o **2º campo** como classe.

### Saída do dataset
- Imagem: carregada em *grayscale* (`convert('L')`) e convertida para tensor com `ToTensor()`.
- Label: convertida para **one-hot** com 10 posições (classes 0–9).

> ⚠️ **Observação:** neste momento o dataset é reduzido a **10%** (via `len(self.images) * 0.1`). Para usar o MNIST completo, esta redução deve ser removida/ajustada.

---

## 3.3 Modelo (`ModelBetterCNN` em `model.py`)
A arquitetura está dividida em dois blocos:

### a) Extração de características (*features*)
- Conv(1→32) + BatchNorm + ReLU  
- Conv(32→32) + BatchNorm + ReLU  
- MaxPool (28→14) + Dropout2d  
- Conv(32→64) + BatchNorm + ReLU  
- Conv(64→64) + BatchNorm + ReLU  
- MaxPool (14→7) + Dropout2d  

### b) Classificador (*classifier*)
- Flatten  
- Linear(64·7·7 → 256) + BatchNorm1d + ReLU + Dropout  
- Linear(256 → 10)

---

## 3.4 Treino e checkpoints (`trainer.py`)
### Treino
- `DataLoader` de treino com `shuffle=True` e teste com `shuffle=False`.
- Otimizador: **Adam** (lr=0.001).
- Loss: `MSELoss(softmax(logits), one-hot)`.

Em cada época:
- Ciclo de treino (atualiza pesos)
- Ciclo de teste (mede loss)
- Guarda histórico de losses e gera gráfico.

### Outputs típicos
- `checkpoint.pkl` — estado completo do treino (época, losses, modelo, otimizador).
- `best.pkl` — melhor modelo (quando a loss de teste melhora).
- `training.png` — gráfico de loss (treino vs teste) ao longo das épocas.

---

## 3.5 Avaliação (`trainer.py`)
Na avaliação:
- Calcula **matriz de confusão** (sklearn).
- Calcula **accuracy**, e **precision/recall/F1 por classe** + **média macro**.
- Guarda:
  - `statistics.json` (métricas por classe + médias)
  - `confusion_matrix.png`
  - `results_table.png`

---

## 3.6 Resultados (espaço para imagens)
> Substitui os caminhos abaixo pelos que existirem na tua pasta de `experiments/`.

### Curvas de treino (loss)
![Training curve](./Tarefa1/experiments/training.png)

### Matriz de confusão
![Confusion matrix](./Tarefa1/experiments/confusion_matrix.png)

### Tabela de métricas (Accuracy/Precision/Recall/F1)
![Results table](./Tarefa1/experiments/results_table.png)

---

# 4) Tarefa 2 — MNIST-Detection (Geração do dataset)

## 4.1 Objetivo
Criar imagens “cena” (ex.: 128×128) com:
- 1 ou vários dígitos por imagem,
- controlo de escala do dígito,
- **sem sobreposição** entre *bounding boxes*,
- labels em `.txt` com as BBs.

---

## 4.2 Geração (`generate_data.py`)
### Como as imagens são criadas
Para cada imagem:
1. Cria um canvas `im` (zeros) com dimensão `imsize×imsize`.
2. Sorteia `n_digits` entre `min_digits_per_image` e `max_digits_per_image`.
3. Para cada dígito:
   - escolhe tamanho entre `min_digit_size` e `max_digit_size`
   - escolhe posição aleatória
   - tenta colocar sem sobreposição usando IoU (aceita apenas se `max(IoU) == 0`)
   - ajusta a bbox ao “conteúdo” do dígito (*tight bbox*)
   - cola no canvas com `np.maximum`

### Formato das labels
Cada imagem tem um `.txt` com:
