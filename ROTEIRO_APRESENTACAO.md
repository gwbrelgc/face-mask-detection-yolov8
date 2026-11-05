# Roteiro de Apresentacao — Face Mask Detection (5 minutos, 4 pessoas)

---

## PESSOA 1 — Problema e Dataset (~1:15)

### O que falar:

"Nosso projeto e a deteccao automatica de mascaras faciais usando deep learning.
O modelo recebe uma imagem e precisa fazer duas coisas ao mesmo tempo: localizar
cada rosto com um bounding box e classificar em 3 categorias — com mascara,
sem mascara, ou mascara incorreta (abaixo do nariz).

Usamos o dataset Face Mask Detection do Kaggle. Sao 853 imagens com 4.072
anotacoes no formato Pascal VOC (XML com coordenadas de cada rosto). Dividimos
80/20 — 682 pra treino, 171 pra validacao.

Um ponto importante: o dataset e desbalanceado. A classe 'com mascara' tem
3.232 amostras (79%), 'sem mascara' tem 717 (18%), mas 'mascara incorreta' tem
so 123 (3%). Isso afeta diretamente o desempenho nessa classe, como vamos ver
nos resultados."

### Pontos-chave:
- Tarefa: Object Detection (nao e classificacao simples — localiza E classifica)
- Dataset: Kaggle, 853 imgs, 3 classes, Pascal VOC → convertido pra YOLO format
- Split: 80/20 aleatorio com seed fixa (reprodutivel)
- Desbalanceamento: 79% / 18% / 3%

---

## PESSOA 2 — Metodologia e Arquitetura (~1:15)

### O que falar:

"Pra escolher a configuracao do modelo, usamos uma metodologia de variavel
controlada. Definimos uma configuracao baseline usando os defaults do YOLOv8 —
modelo Nano, resolucao 640, learning rate 0.01, mosaic augmentation ativo — e
rodamos 6 experimentos variando um parametro por vez.

Testamos: modelo maior (YOLOv8 Small), resolucoes menores (416 e 320), learning
rate mais baixa (0.001), e desativar o mosaic augmentation. Cada experimento
roda com 15 epochs no mesmo hardware pra comparacao justa.

A arquitetura e o YOLOv8 Nano — um detector single-stage que localiza e
classifica em uma unica passada. Tem 3 milhoes de parametros, usa backbone com
blocos C2f e SPPF, neck com Feature Pyramid Network, e 3 cabecas de deteccao
em escalas diferentes. Usamos transfer learning a partir de pesos pre-treinados
no COCO (330 mil imagens), transferindo 349 das 355 camadas."

### Pontos-chave:
- Metodologia: variavel controlada — muda 1 coisa por vez
- 6 experimentos + modelo final
- YOLOv8 Nano: 3M params, single-stage, transfer learning do COCO
- Augmentation: mosaic, flip horizontal, escala +-50%, HSV, erasing

---

## PESSOA 3 — Resultados Experimentais (~1:15)

### O que falar:

"Os resultados dos 6 experimentos mostraram que a configuracao baseline ja era
a melhor. O YOLOv8 Small, com 3 vezes mais parametros, deu 84.8% de mAP contra
85.5% do Nano — ou seja, modelo maior nao melhorou. Isso faz sentido porque com
apenas 853 imagens, um modelo maior tende a overfitting.

A resolucao foi o fator mais critico: baixar de 640 pra 416 perdeu 7% de mAP,
e pra 320 perdeu 14%. O mosaic augmentation contribuiu +1% de mAP. O learning
rate 0.001 convergiu ao mesmo ponto que 0.01.

Com a melhor configuracao definida, treinamos o modelo final com 30 epochs.
Resultado: 85% mAP@50, 59.1% mAP@50-95, 87% precision e 78.1% recall. Por
classe: com mascara 95% AP, sem mascara 85% AP, mascara incorreta 75.1% AP."

### Pontos-chave:
- Nano >= Small neste dataset (justificativa com dados)
- Resolucao e o parametro mais impactante
- Modelo final: 85% mAP@50, treinado com 30 epochs
- Classe dificil (incorreta): 75.1% AP — melhor que o esperado dado o desbalanceamento

---

## PESSOA 4 — Analise de Erros e Conclusao (~1:15)

### O que falar:

"Analisando a matriz de confusao, o principal erro do modelo e confundir mascara
incorreta com mascara correta. Isso e esperado — visualmente a diferenca e so a
posicao da mascara (abaixo do nariz vs cobrindo), e a classe tem apenas 123
amostras no dataset.

As curvas de treinamento mostram que o modelo convergiu sem overfitting — a
validation loss acompanha a training loss ao longo das 30 epochs. O recall subiu
7.5% com mais epochs, mostrando que o modelo final detecta mais rostos.

Como melhorias futuras: oversampling da classe minoritaria com augmentation
especifico, focal loss pra dar mais peso a classes raras, coletar mais dados de
mascara incorreta, e aplicar o modelo em video em tempo real — que e viavel ja
que a inferencia e rapida na GPU.

Em resumo: metodologia com 6 experimentos justificando cada escolha, modelo final
com 85% mAP, e pipeline completamente reprodutivel."

### Pontos-chave:
- Erro principal: incorreta confundida com correta (visual similar + poucos dados)
- Sem overfitting (val loss ok)
- Melhorias: oversampling, focal loss, mais dados, video real-time
- Fechamento: metodologia solida, resultados reprodutiveis

---

## PERGUNTAS ESPERADAS DO PROFESSOR

### 1. "Por que YOLOv8 e nao outro modelo (Faster R-CNN, SSD, etc.)?"

**Resposta:** "YOLO e um detector single-stage — faz localizacao e classificacao
em uma passada so, o que o torna muito mais rapido que two-stage como Faster
R-CNN. Pra uma aplicacao de monitoramento em tempo real, velocidade e essencial.
Alem disso, YOLOv8 e a versao mais recente com melhorias como anchor-free
detection e C2f blocks. O modelo Nano especificamente foi escolhido porque
testamos experimentalmente contra o Small e ele nao trouxe ganho com este
dataset."

### 2. "Por que Nano e nao um modelo maior?"

**Resposta:** "Testamos isso no experimento E1. O YOLOv8 Small tem 11.2 milhoes
de parametros (3x mais que o Nano) e deu 84.8% mAP contra 85.5% do Nano. Com
apenas 853 imagens, o modelo maior nao tem dados suficientes pra aproveitar a
capacidade extra — tende a overfitting. O Nano com transfer learning ja captura
o necessario."

### 3. "Por que resolucao 640 e nao menor?"

**Resposta:** "Testamos 3 resolucoes: 640, 416 e 320. A queda de 640 pra 416 foi
de 7% no mAP, e pra 320 foi de 14%. A resolucao e o parametro que mais impacta
porque muitos rostos no dataset sao pequenos — com resolucao baixa, o modelo
perde detalhe dos bounding boxes menores."

### 4. "Voces usaram so o dataset original? Nao coletaram mais dados?"

**Resposta:** "Sim, usamos apenas o dataset original do Kaggle com 853 imagens.
Porem, o YOLOv8 aplica data augmentation automatico durante o treino — mosaic
(combina 4 imagens), flip horizontal, variacao de escala e cor. Testamos o
impacto do mosaic no experimento E5 e ele contribuiu +1% de mAP. Nao coletamos
dados extras, mas e uma das melhorias propostas, especialmente pra classe
minoritaria."

### 5. "Como voces sabem que nao tem overfitting?"

**Resposta:** "Pelas curvas de treinamento: a validation loss desce junto com a
training loss e nao sobe nas ultimas epochs. Alem disso, usamos early stopping
com patience de 10 epochs — se a validacao parasse de melhorar, o treino parava.
O transfer learning tambem ajuda a prevenir overfitting porque os pesos ja vem
pre-treinados do COCO."

### 6. "Por que 80/20 e nao cross-validation?"

**Resposta:** "Em object detection, cross-validation e pouco usado porque o custo
de treino e muito alto — cada fold e um treino completo. Com 6 experimentos +
modelo final ja foram 7 treinos. A divisao 80/20 com seed fixa garante
reprodutibilidade, e 171 imagens de validacao sao suficientes pra avaliar
metricas estaveis."

### 7. "O que e mAP e por que e a metrica principal?"

**Resposta:** "mAP e Mean Average Precision — calcula a area sob a curva
precision-recall pra cada classe e faz a media. O @0.5 significa que uma deteccao
e considerada correta se o IoU (interseccao sobre uniao) entre o box predito e o
ground truth for maior que 50%. E a metrica padrao em deteccao de objetos porque
avalia tanto a qualidade da localizacao quanto da classificacao."

### 8. "Por que a classe 'mascara incorreta' tem desempenho pior?"

**Resposta:** "Dois motivos. Primeiro, desbalanceamento: so tem 123 amostras (3%
do dataset) contra 3.232 de 'com mascara'. Segundo, ambiguidade visual: mascara
incorreta (abaixo do nariz) parece muito com mascara correta — a diferenca e
sutil, especialmente em rostos pequenos ou distantes."

### 9. "Voces tentaram resolver o desbalanceamento?"

**Resposta:** "O mosaic augmentation ajuda indiretamente porque combina 4 imagens,
aumentando a chance de amostras da classe rara aparecerem. Mas nao fizemos
oversampling especifico nem class weighting — isso ficou como melhoria proposta.
Mesmo assim, a classe minoritaria atingiu 75.1% AP, o que e razoavel dado o
contexto."

### 10. "Quanto tempo levou pra treinar? Isso rodaria em tempo real?"

**Resposta:** "Na GPU (RTX 4060), cada experimento de 15 epochs levou 2-4 minutos.
O modelo final com 30 epochs levou cerca de 7 minutos. A inferencia e muito
rapida — na GPU sao menos de 5ms por imagem, ou seja, mais de 200 FPS. Ate em
CPU roda a ~15 FPS, viavel pra monitoramento."

### 11. "O que e transfer learning e por que usaram?"

**Resposta:** "Transfer learning e reaproveitar pesos de um modelo treinado em
outro dataset. O YOLOv8 Nano vem pre-treinado no COCO, que tem 330 mil imagens
e 80 classes. Transferimos 349 das 355 camadas — o modelo ja sabe detectar
bordas, texturas e formas, e so precisa aprender as especificidades das mascaras.
Sem isso, 853 imagens seriam insuficientes."

### 12. "Qual a limitacao principal do trabalho?"

**Resposta:** "O desbalanceamento das classes. Com apenas 3% de amostras de
mascara incorreta, o modelo tem dificuldade nessa classe. Em um cenario real,
essa e justamente a classe mais importante — detectar uso incorreto. Resolver
isso com mais dados e class weighting e a prioridade se fossemos continuar."

---

## DICAS GERAIS

- **Nao leiam slides** — falem olhando pra frente, slides sao suporte visual
- **Cada pessoa transiciona** pro proximo com uma frase de ligacao
- **Controlem o tempo** — 1:15 por pessoa, total 5 min
- **Na hora das perguntas**: quem sabe melhor responde, nao precisa ser quem apresentou
- **Se nao souber**: "Isso ficou como trabalho futuro" e melhor que inventar
