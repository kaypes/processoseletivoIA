👤 Nome Completo: **João Kayque Pereira de Souza**

---

### 1️⃣ Resumo da Arquitetura do Modelo

A CNN implementada tem 2 blocos convolucionais seguidos de um classificador denso:

### 1️⃣ Resumo da Arquitetura do Modelo

CNN implementada em `train_model.py` com a seguinte estrutura:

| Camada | Configuração | Função |
|---|---|---|
| Conv2D | 32 filtros, 3×3, ReLU | Varre a imagem em busca de padrões visuais simples |
| MaxPooling2D | 2×2 | Comprime o mapa de features mantendo os mais relevantes |
| Conv2D | 64 filtros, 3×3, ReLU | Combina os padrões simples em representações mais complexas |
| MaxPooling2D | 2×2 | Segunda compressão espacial |
| Flatten | — | Transforma a matriz em um vetor linear |
| Dense | 64 neurônios, ReLU | Aprende as relações entre as representações extraídas |
| Dropout | 0.3 | Desativa aleatoriamente 30% dos neurônios durante o treino para evitar memorização |
| Dense | 10 neurônios, Softmax | Atribui uma probabilidade a cada dígito possível (0–9) |

Entrada: imagens 28×28×1, normalizadas para [0, 1].

---

### 2️⃣ Bibliotecas Utilizadas

| Biblioteca | Versão | Uso |
|---|---|---|
| `tensorflow` | 2.x | Treinamento, conversão e otimização do modelo |
| `numpy` | 1.x | Manipulação do dataset de calibração |

---

### 3️⃣ Técnica de Otimização do Modelo

Foi aplicada **Full Integer Quantization** via TensorFlow Lite.

A técnica converte tanto os pesos quanto as ativações do modelo de `float32` para `int8`, usando 100 amostras do MNIST como dataset de calibração para determinar os ranges de cada camada. O resultado é um modelo totalmente inteiro, compatível com microcontroladores sem unidade de ponto flutuante.

Vai além da Dynamic Range Quantization, que só converte os pesos e mantém as ativações em float32.

---

### 4️⃣ Resultados Obtidos

| Métrica | Valor |
|---|---|
| Acurácia no conjunto de teste | **98.95%** |
| Tamanho original (`.h5`) | 1468.2 KB |
| Tamanho otimizado (`.tflite`) | 128.1 KB |
| Redução de tamanho | 91.3% |

A acurácia de 98.95% foi atingida em 5 épocas rodando apenas em CPU, dentro das restrições do ambiente de CI.

---

### 5️⃣ Comentários Adicionais

A principal decisão foi escolher Full Integer Quantization em vez de Dynamic Range Quantization. Apesar da necessidade de um dataset de calibração, tem dois benefícios: maior redução de tamanho e compatibilidade total com hardware embarcado sem FPU, como microcontroladores ARM Cortex-M.
