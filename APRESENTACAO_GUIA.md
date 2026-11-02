# Detecção de Máscaras Faciais com YOLOv8
## Guia Completo para Apresentação — Visão Computacional

---

## 1. DEFINIÇÃO DO PROBLEMA

### O que estamos resolvendo?
Criamos um sistema de **detecção de objetos** (object detection) que identifica automaticamente em imagens se as pessoas estão:
- **Com máscara** (`with_mask`)
- **Sem máscara** (`without_mask`)
- **Com máscara usada incorretamente** (`mask_weared_incorrect`) — ex: abaixo do nariz

### Por que isso é relevante?
Durante e após a pandemia de COVID-19, a fiscalização do uso correto de máscaras em locais públicos (aeroportos, hospitais, transporte) se tornou essencial. Um sistema automatizado de câmeras com IA pode:
- Monitorar o cumprimento de normas sanitárias em tempo real
- Reduzir a necessidade de fiscalização humana
- Emitir alertas automáticos quando alguém não usa máscara

### Tipo de tarefa
Não é uma simples **classificação** (que diria apenas "esta imagem tem máscara ou não"). É **detecção de objetos**: o modelo precisa:
1. **Localizar** cada rosto na imagem (encontrar a posição exata com um retângulo — bounding box)
2. **Classificar** cada rosto detectado em uma das 3 classes

Uma mesma imagem pode ter várias pessoas, cada uma com uma classe diferente.

---

## 2. DATASET

### Origem
- **Fonte:** Kaggle — Face Mask Detection Dataset
- **Total:** 853 imagens com 4.072 anotações (bounding boxes)

### Formato das anotações
As anotações vieram em formato **Pascal VOC** (XML). Cada arquivo XML contém:
```xml
<annotation>
    <filename>imagem.png</filename>
    <size>
        <width>512</width>
        <height>366</height>
    </size>
    <object>
        <name>with_mask</name>       <!-- classe -->
        <bndbox>
            <xmin>79</xmin>          <!-- coordenadas do retângulo -->
            <ymin>105</ymin>
            <xmax>109</xmax>
            <ymax>142</ymax>
        </bndbox>
    </object>
</annotation>
```

### Distribuição das classes
| Classe | Quantidade | Percentual |
|--------|-----------|------------|
| `with_mask` | 3.232 | 79,4% |
| `without_mask` | 717 | 17,6% |
| `mask_weared_incorrect` | 123 | 3,0% |

**Problema identificado:** Existe um forte **desbalanceamento de classes**. A classe `mask_weared_incorrect` tem apenas 123 amostras — isso explica por que o modelo tem mais dificuldade nessa classe.

### Divisão dos dados
- **Treino:** 80% → 682 imagens (para o modelo aprender)
- **Validação:** 20% → 171 imagens (para avaliar o modelo em dados nunca vistos)

A divisão foi feita de forma aleatória com seed fixa (42) para reprodutibilidade.

---

## 3. EXPLICAÇÃO DO CÓDIGO — PASSO A PASSO

### Arquivo `face_mask_detection.py` — Preparação e EDA

#### Função `parse_voc_xml(xml_path)`
**O que faz:** Lê um arquivo XML de anotação e converte as coordenadas para o formato YOLO.

**Conversão de formato:**
- **Pascal VOC:** usa coordenadas absolutas (xmin, ymin, xmax, ymax em pixels)
- **YOLO:** usa coordenadas normalizadas (centro_x, centro_y, largura, altura — valores entre 0 e 1)

**Fórmula de conversão:**
```
centro_x = (xmin + xmax) / 2 / largura_imagem
centro_y = (ymin + ymax) / 2 / altura_imagem
largura_box = (xmax - xmin) / largura_imagem
altura_box = (ymax - ymin) / altura_imagem
```

**Por que normalizar?** Porque o YOLO redimensiona as imagens para um tamanho fixo (416×416). Coordenadas normalizadas (0 a 1) funcionam independente do tamanho original da imagem.

#### Função `prepare_dataset()`
**O que faz:**
1. Cria a estrutura de pastas que o YOLO espera:
   ```
   dataset/
   ├── train/
   │   ├── images/    (imagens de treino)
   │   └── labels/    (arquivos .txt com anotações YOLO)
   ├── val/
   │   ├── images/    (imagens de validação)
   │   └── labels/    (arquivos .txt com anotações YOLO)
   └── data.yaml      (configuração do dataset)
   ```
2. Embaralha as imagens aleatoriamente
3. Separa 80% para treino e 20% para validação
4. Para cada imagem: copia para a pasta correta e gera o arquivo `.txt` com as anotações YOLO

**Formato do arquivo .txt (YOLO):**
```
0 0.183594 0.337158 0.058594 0.101093    (classe centro_x centro_y largura altura)
1 0.401367 0.333333 0.080078 0.120219
```
Classe 0 = with_mask, Classe 1 = without_mask, Classe 2 = mask_weared_incorrect

#### Função `create_data_yaml()`
**O que faz:** Cria o arquivo de configuração que o YOLOv8 precisa para saber:
- Onde estão as imagens de treino e validação
- Quantas classes existem (3)
- Quais são os nomes das classes

#### Funções de visualização (EDA — Exploratory Data Analysis)
- `plot_class_distribution()` — Gráfico de barras mostrando quantas anotações de cada classe existem no treino e validação
- `plot_sample_images()` — Mostra 6 imagens do dataset com os bounding boxes desenhados e coloridos por classe
- `plot_bbox_analysis()` — Histogramas do tamanho dos bounding boxes (largura, altura, e scatter width vs height)

**Por que fazer EDA?** É essencial entender os dados antes de treinar:
- Identificamos o desbalanceamento de classes
- Vimos que a maioria dos bounding boxes são pequenos (rostos distantes)
- Isso ajuda a justificar escolhas de arquitetura e hiperparâmetros

---

### Arquivo `train_model.py` — Treinamento e Avaliação

#### Função `train()`
**O que faz:** Treina o modelo YOLOv8 Nano no nosso dataset.

**Hiperparâmetros escolhidos:**
| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| `model` | YOLOv8n (Nano) | Modelo leve, rápido, bom para datasets pequenos |
| `epochs` | 15 | Suficiente para convergência com early stopping |
| `imgsz` | 416 | Resolução balanceada entre velocidade e precisão |
| `batch` | 8 | Compatível com CPU (sem GPU) |
| `patience` | 5 | Para treino se mAP não melhorar em 5 epochs seguidas |
| `cache` | True | Carrega imagens na RAM para treinar mais rápido |

**Transfer Learning (Aprendizado por Transferência):**
O modelo **NÃO** foi treinado do zero. Usamos pesos pré-treinados no dataset COCO (330K imagens, 80 classes de objetos). Isso é essencial porque:
- Nosso dataset tem apenas 853 imagens — insuficiente para treinar uma rede profunda do zero
- As camadas iniciais da rede já aprenderam a detectar bordas, texturas e formas genéricas
- Apenas ajustamos (fine-tuning) as camadas finais para nossa tarefa específica

**Data Augmentation (Aumento de Dados):**
O YOLOv8 aplica automaticamente diversas transformações durante o treino:
- **Mosaic:** combina 4 imagens em uma só (ajuda com objetos pequenos e contextos variados)
- **Flip horizontal:** espelha a imagem (50% de chance)
- **Scale:** zoom aleatório (±50%)
- **HSV:** variação de matiz, saturação e brilho
- **Erasing:** apaga retângulos aleatórios da imagem (40%)

Isso aumenta artificialmente a diversidade dos dados e ajuda a prevenir overfitting.

#### Função `evaluate()`
**O que faz:** Avalia o melhor modelo salvo (`best.pt`) no conjunto de validação.

**Métricas calculadas:**
- **Precision (Precisão):** Das detecções que o modelo fez, quantas estavam corretas?
  - `Precision = Verdadeiros Positivos / (Verdadeiros Positivos + Falsos Positivos)`
  - Alta precisão = poucas detecções falsas

- **Recall (Revocação):** De todos os objetos reais, quantos o modelo encontrou?
  - `Recall = Verdadeiros Positivos / (Verdadeiros Positivos + Falsos Negativos)`
  - Alto recall = poucas detecções perdidas

- **mAP@0.5 (mean Average Precision):** Média da precisão para cada classe, considerando uma detecção correta quando o IoU (Intersection over Union) entre o box previsto e o real é ≥ 50%

- **mAP@0.5:0.95:** Média do mAP calculado em múltiplos limiares de IoU (de 0.5 a 0.95, incremento de 0.05). É a métrica mais rigorosa — exige boxes muito precisos.

**O que é IoU (Intersection over Union)?**
```
IoU = Área de Interseção / Área de União
```
Mede o quanto o bounding box previsto pelo modelo se sobrepõe ao bounding box real (ground truth). IoU = 1.0 significa sobreposição perfeita.

#### Função `inference()`
**O que faz:** Seleciona 9 imagens aleatórias da validação e roda o modelo para gerar predições visuais com os bounding boxes, classes e scores de confiança desenhados na imagem.

---

## 4. ARQUITETURA DO MODELO — YOLOv8

### O que é YOLO?
**YOLO** = "You Only Look Once" (Você Só Olha Uma Vez)

Diferente de outros detectores que analisam a imagem em múltiplas etapas, o YOLO faz tudo em **uma única passada pela rede neural:**
1. Divide a imagem em uma grade (grid)
2. Para cada célula da grade, prediz bounding boxes e probabilidades de classe
3. Filtra as detecções com NMS (Non-Maximum Suppression) para eliminar duplicatas

**Vantagem:** Velocidade (detecção em tempo real)

### YOLOv8 Nano — Estrutura
O modelo tem **3.011.433 parâmetros** e **129 camadas**:

1. **Backbone (Extrator de features):**
   - Camadas convolucionais (Conv) que extraem características da imagem
   - Blocos C2f (Cross Stage Partial com 2 convoluções) — eficientes em extrair features multi-escala
   - SPPF (Spatial Pyramid Pooling Fast) — captura contexto em diferentes escalas

2. **Neck (Fusão de features):**
   - FPN (Feature Pyramid Network) — combina features de diferentes resoluções
   - Upsample + Concatenate — mistura features detalhadas (resolução alta) com features semânticas (resolução baixa)

3. **Head (Cabeça de detecção):**
   - Detect — gera as predições finais em 3 escalas diferentes
   - Cada escala detecta objetos de tamanhos diferentes (pequenos, médios, grandes)

### Por que YOLOv8 Nano?
| Variante | Parâmetros | GFLOPs | mAP (COCO) |
|----------|-----------|--------|------------|
| YOLOv8n (Nano) | 3.0M | 8.2 | 37.3 |
| YOLOv8s (Small) | 11.2M | 28.6 | 44.9 |
| YOLOv8m (Medium) | 25.9M | 78.9 | 50.2 |
| YOLOv8l (Large) | 43.7M | 165.2 | 52.9 |

Escolhemos o Nano porque:
- Treinamento em CPU — modelos maiores seriam muito lentos
- Dataset pequeno (853 imagens) — modelo maior poderia causar overfitting
- Ainda assim alcança bons resultados com transfer learning

---

## 5. RESULTADOS OBTIDOS

### Métricas Gerais
| Métrica | Valor |
|---------|-------|
| **mAP@0.5** | **0.745** (74.5%) |
| **mAP@0.5:0.95** | **0.486** (48.6%) |
| **Precision** | **0.762** (76.2%) |
| **Recall** | **0.714** (71.4%) |

### Métricas Por Classe
| Classe | AP@0.5 | Precision | Recall |
|--------|--------|-----------|--------|
| `with_mask` | **0.932** | 0.89 | 0.90 |
| `without_mask` | **0.751** | 0.79 | 0.69 |
| `mask_weared_incorrect` | **0.552** | 0.60 | 0.56 |

### Análise da Matriz de Confusão
- **with_mask:** 96% de acerto — excelente! O modelo quase nunca confunde quem está de máscara.
- **without_mask:** 83% de acerto — bom resultado. 9% das vezes confunde com with_mask.
- **mask_weared_incorrect:** 41% de acerto — a classe mais difícil. 50% das vezes é confundida com with_mask.

### Interpretação das Curvas
- **Curvas de loss (treino e validação):** Todas decrescem continuamente ao longo das 15 epochs, indicando que o modelo está aprendendo sem overfitting significativo.
- **Curva Precision-Recall:** A classe `with_mask` tem área sob a curva quase perfeita (0.933). A `mask_weared_incorrect` é a mais fraca (0.555).
- **Curva F1-Confidence:** O melhor F1 geral (0.73) é alcançado com threshold de confiança 0.266.

---

## 6. ANÁLISE CRÍTICA

### Pontos fortes
1. **Alta performance em `with_mask`** (93.2% AP) — a classe mais importante está muito bem coberta
2. **Transfer learning eficaz** — com apenas 853 imagens e 15 epochs, alcançamos 74.5% mAP
3. **Tempo de inferência rápido** — ~67ms por imagem (mesmo em CPU), viável para aplicações em tempo real com GPU
4. **Pipeline completo e reprodutível** — da preparação dos dados até a avaliação final

### Limitações e desafios
1. **Desbalanceamento de classes** — `mask_weared_incorrect` tem apenas 123 amostras (3% do total), resultando em baixo desempenho nessa classe
2. **Treinamento em CPU** — limitou o número de epochs e tamanho da imagem. Com GPU, poderíamos treinar mais tempo com resolução maior
3. **Dataset limitado** — 853 imagens é relativamente pequeno para detecção de objetos
4. **Confusão entre classes similares** — `mask_weared_incorrect` é visualmente muito parecida com `with_mask`, dificultando a distinção

### Possíveis melhorias
1. **Oversampling / Data augmentation extra** para a classe minoritária (`mask_weared_incorrect`)
2. **Treinar com GPU** — mais epochs (50-100), imagem maior (640px), modelo maior (YOLOv8s)
3. **Usar focal loss** com peso maior para classes raras
4. **Aumentar o dataset** — coletar mais imagens de máscara incorreta
5. **Ensemble de modelos** — combinar predições de múltiplos modelos
6. **Teste em vídeo** — aplicar o modelo em um stream de câmera para demonstração em tempo real

---

## 7. GLOSSÁRIO — TERMOS PARA A APRESENTAÇÃO

| Termo | Explicação simples |
|-------|-------------------|
| **CNN** | Rede Neural Convolucional — tipo de rede que "varre" a imagem com filtros para encontrar padrões |
| **YOLO** | You Only Look Once — arquitetura que detecta objetos em uma única passada |
| **Transfer Learning** | Reaproveitar conhecimento de um modelo pré-treinado em outro dataset |
| **Fine-tuning** | Ajustar as camadas finais de um modelo pré-treinado para uma tarefa nova |
| **Bounding Box** | Retângulo que delimita um objeto detectado na imagem |
| **IoU** | Intersection over Union — mede sobreposição entre box previsto e real |
| **mAP** | mean Average Precision — métrica padrão para avaliação de detecção de objetos |
| **Precision** | Dos que o modelo detectou, quantos estavam corretos |
| **Recall** | Dos que existiam, quantos o modelo encontrou |
| **F1-Score** | Média harmônica entre Precision e Recall |
| **Overfitting** | Quando o modelo decora os dados de treino e não generaliza |
| **Data Augmentation** | Transformações nas imagens para aumentar diversidade dos dados |
| **NMS** | Non-Maximum Suppression — remove detecções duplicadas |
| **Epoch** | Uma passada completa por todos os dados de treino |
| **Batch** | Grupo de imagens processadas juntas em cada iteração |
| **Backbone** | Parte da rede que extrai features da imagem |
| **Feature Map** | Representação intermediária da imagem dentro da rede |

---

## 8. ESTRUTURA SUGERIDA PARA A APRESENTAÇÃO (10 min)

| Tempo | Slide | Conteúdo |
|-------|-------|----------|
| 0-1 min | 1 | Título + Introdução do problema (por que detectar máscaras?) |
| 1-2 min | 2 | Dataset — origem, tamanho, classes, mostrar exemplos de imagens |
| 2-3 min | 3 | EDA — gráfico de distribuição + análise de desbalanceamento |
| 3-5 min | 4-5 | Metodologia — YOLO (como funciona), transfer learning, hiperparâmetros |
| 5-6 min | 6 | Pipeline do código — fluxo: dados → conversão → treino → avaliação |
| 6-8 min | 7-8 | Resultados — métricas, curvas de treino, matriz de confusão, PR curve |
| 8-9 min | 9 | Predições visuais — mostrar imagens com detecções do modelo |
| 9-10 min | 10 | Análise crítica — limitações, melhorias, conclusão |

---

## 9. PERGUNTAS PROVÁVEIS DO PROFESSOR E RESPOSTAS

**P: Por que escolheram YOLO e não Faster R-CNN ou SSD?**
R: YOLO é mais rápido (detecção em tempo real), mais simples de implementar com a biblioteca Ultralytics, e é o estado da arte para detecção de objetos. Faster R-CNN é mais preciso mas muito mais lento, especialmente sem GPU.

**P: Por que o desempenho na classe `mask_weared_incorrect` é baixo?**
R: Dois fatores: (1) desbalanceamento — apenas 123 amostras vs 3.232 de `with_mask`; (2) similaridade visual — uma máscara abaixo do nariz é visualmente muito parecida com uma máscara correta, dificultando a distinção.

**P: O que é transfer learning e por que usaram?**
R: É reaproveitar um modelo que já aprendeu features genéricas (bordas, texturas, formas) em um dataset grande (COCO, 330K imagens). Com apenas 853 imagens seria impossível treinar uma rede profunda do zero — não haveria dados suficientes para aprender features básicas.

**P: Como vocês evitaram overfitting?**
R: (1) Transfer learning — partimos de features já aprendidas; (2) Data augmentation — mosaic, flip, scale, HSV; (3) Early stopping — paramos se não melhorar em 5 epochs; (4) Modelo Nano — menos parâmetros = menos risco de decorar dados.

**P: O que fariam diferente com mais tempo/recursos?**
R: Treinar com GPU usando mais epochs e resolução maior, usar modelo YOLOv8s ou m, aplicar oversampling na classe minoritária, e testar em vídeo em tempo real.

**P: Qual a diferença entre mAP@0.5 e mAP@0.5:0.95?**
R: mAP@0.5 considera uma detecção correta se o IoU é ≥ 50%. mAP@0.5:0.95 calcula a média do mAP em vários limiares (50%, 55%, ..., 95%), sendo muito mais rigoroso — exige que os bounding boxes sejam muito precisos.

**P: O modelo funciona em tempo real?**
R: Em CPU, ~67ms por imagem (~15 FPS). Com uma GPU, o YOLOv8 Nano alcança facilmente 100+ FPS, viável para câmeras em tempo real.

---

## 10. ARQUIVOS DO PROJETO

```
F:\Visao\
├── images/                    # 853 imagens originais (.png)
├── annotations/               # 853 anotações Pascal VOC (.xml)
├── dataset/
│   ├── data.yaml              # Configuração do dataset para YOLO
│   ├── train/
│   │   ├── images/            # 682 imagens de treino
│   │   └── labels/            # 682 labels YOLO (.txt)
│   └── val/
│       ├── images/            # 171 imagens de validação
│       └── labels/            # 171 labels YOLO (.txt)
├── results/
│   ├── class_distribution.png # Gráfico de distribuição das classes
│   ├── sample_images.png      # Exemplos anotados do dataset
│   ├── bbox_analysis.png      # Análise de tamanho dos bboxes
│   ├── report.txt             # Relatório final em texto
│   ├── train/
│   │   ├── weights/
│   │   │   ├── best.pt        # Melhor modelo (usado para avaliação)
│   │   │   └── last.pt        # Último modelo salvo
│   │   ├── results.png        # Curvas de treino completas
│   │   ├── confusion_matrix_normalized.png
│   │   ├── BoxPR_curve.png    # Curva Precision-Recall
│   │   ├── BoxF1_curve.png    # Curva F1 vs Confidence
│   │   └── ...
│   ├── val/                   # Plots de validação
│   └── predictions/           # 9 imagens com predições visuais
├── face_mask_detection.py     # Script de preparação + EDA
├── train_model.py             # Script de treino + avaliação
└── Visao.txt                  # Enunciado do trabalho
```
