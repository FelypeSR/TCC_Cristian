# Detecção de Smishing em Idosos com Modelos de Linguagem Natural

**Trabalho de Conclusão de Curso II — Bacharelado em Ciência da Computação**
Instituto Federal de Educação, Ciência e Tecnologia do Maranhão — Campus Caxias
Autor: Cristian Vitor Alves Lopes
Orientador: Prof. Dr. Paulo Henrique Franco Rocha

---

## Visão Geral

Estudo comparativo da aplicabilidade de modelos de linguagem natural na detecção automatizada de mensagens de smishing (SMS phishing) direcionadas à população idosa brasileira. O projeto avalia três trilhas de modelagem sobre o mesmo conjunto de teste, produzindo métricas comparáveis e diretrizes técnicas para soluções de segurança digital acessíveis.

| Trilha | Modelos | Papel |
|---|---|---|
| Clássica | Complement Naive Bayes, SVM linear | Piso de desempenho, interpretável e barato |
| Fine-tuning | BERTimbau | Modelo pré-treinado em português ajustado à tarefa |
| Prompting | Llama 3.2-3B-Instruct (4-bit) + rubrica | Classificação por rubrica com pontuação de risco |

> **Coerência com a monografia.** Este documento implementa a metodologia descrita nas seções 4.3 e 4.4 do TCC. Alguns pontos aqui **exigem edição correspondente no texto** da monografia — todos estão listados na seção [Ajustes necessários no texto do TCC](#ajustes-necessários-no-texto-do-tcc), no final. Implementar sem fazer esses ajustes cria divergência entre o código e o documento aprovado.

---

## Estrutura do Projeto

O código vive em repositório Git. O Google Drive guarda **apenas dados e resultados** — nunca código.

```
tcc-smishing/                      # repositório Git
├── notebooks/
│   ├── 00_augmentation.ipynb      # uso único (GPU) — gera mensagens sintéticas
│   ├── 01_dados.ipynb             # CPU
│   ├── 02_baseline_classico.ipynb # CPU
│   ├── 03_bertimbau.ipynb         # GPU
│   ├── 04_llm_llama.ipynb         # GPU
│   └── 05_avaliacao.ipynb         # CPU
│
├── src/
│   ├── config.py                  # seed, rótulos, paths, constantes
│   ├── preprocessing.py           # limpeza, tokenização, normalização
│   ├── rubrica.py                 # critérios e prompt da rubrica de risco
│   └── evaluation.py              # métricas, matriz de confusão, figuras
│
├── requirements.txt
└── README.md
```

```
/content/drive/MyDrive/TCC_Smishing/    # Google Drive — só dados e saídas
├── data/
│   ├── raw/            # fontes originais
│   ├── synthetic/      # mensagens geradas (saída do notebook 00)
│   ├── processed/      # corpus consolidado e rotulado
│   └── splits/         # train / val / test fixos + split_info.json
└── results/
    ├── predictions/    # uma saída padronizada por modelo
    ├── metrics/        # tabelas consolidadas
    ├── figures/        # matrizes de confusão, ROC, comparativos
    └── models/         # modelo final do BERTimbau
```

**Por que o código não fica no Drive:** módulo importado do Drive fica cacheado pelo Python — editar `preprocessing.py` no meio da sessão não tem efeito até reiniciar o runtime, e isso custa horas de depuração fantasma. Pelo Git, cada notebook puxa a versão corrente e o histórico fica versionado, o que também sustenta o argumento de reprodutibilidade usado para justificar a troca de GPT por Llama.

---

## Ambiente e Setup

**Importante sobre o Colab:** cada notebook roda em um runtime independente. Instalar dependências em um notebook **não** vale para os outros. Por isso a célula de setup se repete em todos, e não existe um "notebook de instalação".

Célula padrão no topo de todos os notebooks:

```python
# 1. Clona/atualiza o código do repositório
!git clone -q https://github.com/FelypeSR/TCC_Cristian.git /content/TCC_Cristian \
    2>/dev/null || (cd /content/TCC_Cristian && git pull -q)

# 2. Instala as dependências com versões fixas
!pip install -q -r /content/TCC_Cristian/requirements.txt

# 3. Monta o Drive (dados e resultados)
from google.colab import drive
drive.mount('/content/drive')

# 4. Importa a configuração compartilhada
import sys
sys.path.insert(0, '/content/TCC_Cristian/src')
import config as CFG
```

### `requirements.txt`

```
scikit-learn>=1.3
pandas
numpy
matplotlib
seaborn
nltk
spacy
ucimlrepo
transformers>=4.41        # eval_strategy; em versões anteriores era evaluation_strategy
datasets
accelerate
bitsandbytes
sentencepiece
protobuf
statsmodels               # teste de McNemar (opcional)
```

> Depois da primeira execução bem-sucedida, rode `pip freeze` e **fixe as versões exatas** no arquivo. `transformers` e `bitsandbytes` quebram entre versões com frequência; sem pinagem, o experimento não é replicável e o argumento de reprodutibilidade da seção 4.3.3 fica sem apoio.

O modelo do spaCy precisa ser baixado a cada runtime que use pré-processamento:

```python
!python -m spacy download pt_core_news_sm -q
```

### `src/config.py`

Centraliza o que hoje seria redigitado em cinco notebooks:

```python
SEED = 42
BASE = '/content/drive/MyDrive/TCC_Smishing'

CLASSE_POSITIVA = 'smishing'
CLASSE_NEGATIVA = 'legitima'

SPLIT_RATIO = {'train': 0.70, 'val': 0.15, 'test': 0.15}

PATHS = {
    'raw':         f'{BASE}/data/raw',
    'synthetic':   f'{BASE}/data/synthetic',
    'processed':   f'{BASE}/data/processed',
    'splits':      f'{BASE}/data/splits',
    'predictions': f'{BASE}/results/predictions',
    'metrics':     f'{BASE}/results/metrics',
    'figures':     f'{BASE}/results/figures',
    'models':      f'{BASE}/results/models',
}

MODELOS = {
    'bertimbau': 'neuralmind/bert-base-portuguese-cased',
    'llama':     'meta-llama/Llama-3.2-3B-Instruct',
}
```

---

## Notebooks

### `00_augmentation.ipynb` — Geração de Mensagens Sintéticas (GPU, uso único)

**Não faz parte do fluxo recorrente.** Roda uma vez, produz um CSV versionado, e nunca mais é executado. Isso mantém o `01_dados` em CPU de verdade e evita disputar a cota de GPU com os notebooks 03 e 04.

**Respaldo na metodologia:** a seção 4.4.2 prevê "mensagens fraudulentas reais (quando possível) **ou simuladas com base em evidências**", com rotulagem manual.

**Pipeline:**
1. Carrega as mensagens-semente reais da curadoria PT-BR (Bortot et al., CERT.br/GOV.BR)
2. Gera variações controladas por prompt, preservando o tipo de golpe da semente
3. Registra a proveniência: cada mensagem gerada carrega `id_semente` e `tipo_golpe`
4. **Validação manual obrigatória** — revisar mensagem a mensagem, descartar as implausíveis, confirmar o rótulo
5. Salva em `data/synthetic/sinteticas_validadas.csv`

**Saída:** `id, texto, rotulo, tipo_golpe, id_semente, fonte='sintetica', idioma='pt'`

> A coluna `id_semente` não é opcional. É ela que impede que variações da mesma mensagem original caiam em lados opostos do split — o vazamento mais provável e mais destrutivo deste projeto (ver [Regra de split](#regra-de-split-e-vazamento)).

---

### `01_dados.ipynb` — Coleta, Rotulagem e Splits (CPU)

**Objetivo:** construir e persistir a base unificada no Drive.

**Fontes:**

| Fonte | Tipo | Idioma | Papel |
|---|---|---|---|
| SMS Spam Collection (UCI) | Dataset público | Inglês | Validação do pipeline |
| Smishtank.com | Dataset público | Inglês | Referência de padrões (ver nota) |
| Bortot et al. (2024) | Curadoria acadêmica | PT-BR | Semente do corpus principal |
| CERT.br / GOV.BR | Curadoria institucional | PT-BR | Semente do corpus principal |
| Sintéticas validadas | Geração controlada | PT-BR | Expansão do corpus (só treino) |

> **Nota metodológica (SMS Spam Collection):** é spam genérico em inglês, não smishing em português. Seu papel é exclusivamente validar que a esteira de pré-processamento e modelagem funciona mecanicamente. Resultados obtidos nele **não** constituem achados do domínio do TCC e devem aparecer no texto claramente separados.

> **Decidir antes de implementar — Smishtank:** o papel dele precisa ser explícito. Três opções, escolha uma e escreva na 4.3.1: (a) entra no corpus como fonte em inglês, (b) serve apenas para validação de pipeline junto com o SMS Spam Collection, (c) é usado só como referência de padrões para orientar a curadoria e a geração sintética, sem entrar em nenhum treino. Ambiguidade aqui é pergunta certa de banca. **A opção (c) é a mais defensável** dado que o corpus-alvo é PT-BR.

#### Regra de split e vazamento

Esta é a decisão mais importante do projeto inteiro.

**Ordem obrigatória: split primeiro, sintéticas depois — e só no treino.**

```
corpus real → split 70/15/15 estratificado (seed 42)
                    │
                    ├── train ← recebe as mensagens sintéticas
                    ├── val   ← 100% real
                    └── test  ← 100% real
```

Assim val e test contêm apenas mensagens reais, e a monografia pode afirmar que "os modelos foram avaliados exclusivamente sobre mensagens reais". Isso responde de uma vez a duas perguntas de banca: a do vazamento e a de "seu modelo não está apenas reconhecendo o estilo do gerador?".

**Se não houver volume suficiente de mensagens reais** — cenário previsto pela própria 4.4.2 ("quando possível") — a regra mínima inegociável passa a ser: **split por grupo usando `id_semente`**, garantindo que todas as variações de uma mesma semente fiquem do mesmo lado. Nunca um `train_test_split` aleatório sobre um corpus que contém variações da mesma origem: as métricas sobem, os três modelos parecem ótimos, e o resultado não vale nada.

**Pipeline:**
1. Carrega as fontes reais e padroniza o esquema
2. Deduplica por texto normalizado (não por texto exato — variações triviais de espaçamento e pontuação escapam da comparação exata)
3. Split 70/15/15 estratificado por rótulo, `random_state=42`
4. Incorpora as sintéticas validadas **apenas ao treino**
5. Persiste os splits e o registro de proveniência

**Saídas obrigatórias:**
- `data/processed/corpus_completo.csv` — `id, texto, rotulo, tipo_golpe, id_semente, fonte, idioma`
- `data/splits/train.csv`, `val.csv`, `test.csv` — **com todas as colunas acima**, não apenas texto e rótulo
- `data/splits/split_info.json` — seed, proporções, contagem por classe em cada subconjunto, contagem de sintéticas no treino, data de geração

> Os splits precisam carregar `tipo_golpe` e `fonte`, senão a análise qualitativa de erros do notebook 05 não tem de onde tirar o dado.

**Verificações que o notebook deve imprimir:**
- Proporção da classe positiva em train/val/test (a estratificação funcionou?)
- Interseção de `id` entre os três subconjuntos (deve ser vazia)
- Interseção de `id_semente` entre train e test (deve ser vazia)
- Confirmação de que val e test não contêm `fonte == 'sintetica'`

**Aspectos éticos e LGPD** (conforme 4.4.2):
- Anonimização antes de qualquer processamento: mascarar nomes, CPFs, números de telefone e contas
- Mensagens sintéticas com finalidade exclusivamente acadêmica, sem reprodução de dados pessoais de terceiros
- Prioridade para bases públicas ou com licença aberta
- SMS Spam Collection: CC BY 4.0, uso e adaptação permitidos com citação

---

### `02_baseline_classico.ipynb` — Naive Bayes e SVM (CPU)

**Objetivo:** estabelecer o piso de desempenho. Se BERTimbau e Llama não superarem os clássicos de forma relevante, a complexidade adicional não se justifica — e isso é um achado, não um fracasso.

**Pipeline:**
1. Carrega `train.csv`, `val.csv` e `test.csv`
2. Pré-processamento via `src/preprocessing.py` (NLTK + spaCy)
3. Vetorização TF-IDF (unigramas e bigramas) + features artesanais
4. Treina Complement Naive Bayes e SVM linear
5. Calibra o limiar de decisão na **validação**
6. Avalia no teste e grava as predições

**Pré-processamento** (`src/preprocessing.py`, compartilhado):

Conforme as seções 4.3.2.1 e 4.3.2.2, usando NLTK e spaCy:
- Normalização Unicode e de espaçamento
- **Substituição por tokens especiais antes do lowercase:** URLs → `urltoken`, telefones → `telefonetoken`, valores numéricos → `numerotoken`. Substituir em vez de remover preserva a *presença* dessas entidades como sinal — e presença de link é um dos indicadores mais fortes de smishing
- Tokenização (NLTK)
- Lematização (spaCy, `pt_core_news_sm`)
- Remoção de stopwords **como variante experimental, não como padrão** — SMS são curtíssimos e remover stopwords de uma mensagem de 15 palavras descarta boa parte do sinal. Reportar as duas versões e escolher pela validação

**Features artesanais** (complementam o TF-IDF): presença de URL, presença de telefone, termos de urgência, referência a instituição financeira ou órgão público, promessa de prêmio, comprimento da mensagem, proporção de maiúsculas.

**Escolha do Naive Bayes:** `ComplementNB` em vez de `MultinomialNB`. É uma variante do Naive Bayes — portanto coberta pela 4.4.3 — otimizada para classes desbalanceadas, que é exatamente o caso de corpora de spam/smishing. Justificar em uma linha no texto.

**SVM:** `LinearSVC` com `class_weight='balanced'`. Para obter probabilidades (necessárias para AUC e para a calibração de limiar), envolver em `CalibratedClassifierCV` com `method='sigmoid'` — a calibração isotônica superajusta em corpus pequeno e degrada tanto as probabilidades quanto as predições.

**Interpretabilidade:** ao listar os termos mais discriminativos, a ordenação correta no `ComplementNB` é a **decrescente**, igual à do `MultinomialNB`. O nome `feature_log_prob_` sugere pesos do complemento e induz a inverter a ordem, mas na implementação do sklearn, com o padrão `norm=False`, o atributo guarda o *negativo* do log das frequências do complemento e a predição é feita por `argmax` — peso alto significa termo mais associado à classe. O notebook 02 inclui uma célula que confirma isso comparando com o `MultinomialNB` num corpus controlado; rode-a antes de mexer na ordenação.

**Saída:** dois arquivos separados, `results/predictions/naive_bayes.csv` e `results/predictions/svm.csv`.

---

### `03_bertimbau.ipynb` — Fine-tuning BERTimbau (GPU)

**Modelo:** `neuralmind/bert-base-portuguese-cased`

**Texto de entrada:** o texto **original**, não o normalizado. O BERTimbau é *cased* — maiúsculas carregam informação, e o subword tokenizer dele já lida com a variação de superfície que a normalização dos clássicos remove. Essa distinção precisa aparecer no texto: as trilhas usam pré-processamentos diferentes **por decisão fundamentada**, não por descuido.

**Hiperparâmetros:**

| Parâmetro | Valor | Justificativa |
|---|---|---|
| `max_length` | 128 tokens | SMS raramente ultrapassa 160 caracteres |
| Batch size | 16 | Cabe na VRAM da T4 com margem |
| Épocas | 5 (máx.) | Com early stopping |
| Learning rate | 2e-5 | Padrão para fine-tuning de BERT |
| `fp16` | True | Precisão mista — reduz memória e acelera na T4 |
| Seleção do checkpoint | F2 na validação | Coerente com o custo assimétrico do erro |

**Desbalanceamento:** usar `class_weight` na função de perda (via `Trainer` customizado) ou justificar por que não. Sem isso, o modelo tende a favorecer a classe majoritária, o que contraria a priorização do recall.

**Checkpoints — específico do Colab:**
- `output_dir` em `/content/bertimbau_ckpt` (disco local, rápido), **não** no Drive. Cada checkpoint tem ~440 MB; gravar via FUSE a cada época é lento e consome a cota de 15 GB
- `save_total_limit=2`
- `resume_from_checkpoint=True` — sem isso, os checkpoints não protegem de queda de sessão, que é a razão de existirem
- Ao final, copiar **apenas o modelo final** para `results/models/bertimbau_final/`

**Saída:** `results/predictions/bertimbau.csv`

---

### `04_llm_llama.ipynb` — Classificação via Rubrica com Llama (GPU)

**Objetivo:** avaliar classificação few-shot por prompt estruturado em formato de rubrica, no espírito de Dimario, Bacha e Butka (2024).

**Modelo:** `meta-llama/Llama-3.2-3B-Instruct`, quantizado em 4-bit (NF4 + double quant) via `bitsandbytes`.

> **Nota metodológica (GPT → Llama):** o projeto originalmente previa GPT (OpenAI). Por restrições de custo e acessibilidade, optou-se por um modelo aberto (Llama Instruct quantizado), alinhado ao objetivo de avaliar soluções gratuitas e reprodutíveis. Pesos públicos permitem replicação exata do experimento, o que uma API paga não garante. **Esta substituição exige atualizar o resumo, o abstract e a seção 3.3 da monografia**, que ainda citam GPT entre os modelos avaliados.

#### A rubrica

A seção 2 (Justificativa) e a seção 5 do TCC prometem uma rubrica que atribui **pontuação e níveis de risco**, não apenas um rótulo binário. O desenho respeita isso: o modelo avalia critérios explícitos, e o rótulo binário é derivado do score.

Critérios ancorados no referencial teórico do próprio TCC — Bortot et al. (2024, p. 13), citado na seção 3.2, e os padrões descritos na 3.1:

| # | Critério | Origem |
|---|---|---|
| 1 | Falsa identidade institucional (banco, saúde, órgão de governo) | Bortot et al. |
| 2 | Urgência artificial / "verificação imediata" | Bortot et al. |
| 3 | Link ou pedido de clique | Bortot et al. |
| 4 | Promessa de prêmio ou benefício com ação rápida | Bortot et al. |
| 5 | Solicitação de dados pessoais ou bancários | Seção 3.1 |
| 6 | Ameaça de perda (bloqueio de conta, corte de benefício) | Seção 3.2 |
| 7 | Emergência familiar simulada | Seção 3.2 |

O modelo devolve o score de risco; as faixas alimentam a seção de diretrizes (4.4.5):

| Faixa | Nível | Ação sugerida na diretriz |
|---|---|---|
| 0.00 – 0.33 | Baixo | Sem alerta |
| 0.34 – 0.66 | Médio | Alerta informativo com explicação dos indícios |
| 0.67 – 1.00 | Alto | Alerta destacado e orientação de não interagir |

Os níveis são o que conecta este notebook à etapa 4.4.5 e ao discurso educativo que o TCC sustenta do começo ao fim. Sem eles, uma entrega prometida na seção 5 fica sem contrapartida.

**Prevenção de vazamento:** os exemplos few-shot vêm **exclusivamente do treino**, amostrados com a seed do projeto. Nunca de val ou test.

**Uso da validação:** o `val.csv` serve para **escolher o prompt** (zero-shot vs. few-shot, variações da rubrica, número de exemplos). Iterar o prompt olhando o resultado do teste é vazamento pelo conjunto de teste — a versão final do prompt precisa ser escolhida antes de o teste ser tocado.

**Dois pontos que costumam passar despercebidos:**

1. **Verifique os IDs de token antes de rodar o teste completo.** Se o score sair da comparação entre a probabilidade dos tokens de "smishing" e "legitima", confirme que os IDs obtidos pelo tokenizador correspondem ao que o modelo realmente emite — o tokenizador do Llama quebra essas palavras em subtokens e a forma com espaço à frente tem ID diferente. Se não baterem, o score fica praticamente constante, a curva ROC do Llama vira ruído e **nenhum erro é levantado**. Imprima o argmax real do primeiro token gerado em alguns exemplos e compare.

2. **O fallback de resposta inválida deve ir para `smishing`, não para `legitima`.** Quando o modelo não segue o formato, cair no rótulo negativo produz justamente o erro mais caro. Alternativa mais transparente: registrar como "não classificado" e reportar a taxa de abstenção como resultado — é informação honesta sobre a viabilidade da abordagem por prompting.

**Desempenho — específico do Colab:** processar as mensagens em lote (`padding_side='left'`, batches de 8 a 16). Uma a uma, com limite de 90 min de ociosidade e 12 h de sessão, a inferência pode não terminar. Manter também o salvamento periódico de checkpoint para retomar a inferência se a sessão cair.

**Saída:** `results/predictions/llm_llama.csv`, com o score de risco preenchido.

---

### `05_avaliacao.ipynb` — Avaliação Comparativa (CPU)

**Objetivo:** comparar as trilhas em condições unificadas e produzir os artefatos finais.

**Pipeline:**
1. Varre `results/predictions/*.csv` — cada arquivo é uma trilha
2. **Verifica a integridade da comparação:** os `id` avaliados são idênticos entre todos os modelos? Qualquer divergência invalida a comparação e o notebook deve parar
3. Calcula as métricas por modelo
4. Gera matriz de confusão por modelo e comparativo consolidado
5. Análise qualitativa dos erros cruzada com `tipo_golpe`
6. Salva tabelas em `results/metrics/` e figuras em `results/figures/`

**Análise qualitativa** — atende à parte qualitativa declarada na 4.1 e ao que a seção 5 promete:
- Quais tipos de golpe cada modelo mais erra?
- Falsos negativos unânimes: quais mensagens escaparam de todos os modelos? São o material mais rico da discussão
- Falsos positivos unânimes: que característica das mensagens legítimas confunde todos?
- Junta com `test.csv` por `id` para recuperar o texto e o `tipo_golpe`

---

## Contrato de Dados

### Saída padronizada das predições

Um arquivo por modelo, todos com o mesmo esquema:

| Campo | Descrição |
|---|---|
| `id` | Identificador da mensagem, correspondente ao `test.csv` |
| `rotulo_real` | Rótulo verdadeiro |
| `rotulo_predito` | Predição do modelo |
| `score` | Probabilidade / pontuação de risco da classe positiva |
| `modelo` | Identificação do modelo |

O texto **não** é duplicado nas predições — o notebook 05 recupera por `id` a partir do `test.csv`. Isso evita inconsistência e força o `id` a ser chave de verdade.

**Um arquivo por modelo, não por notebook.** O notebook 02 treina dois modelos e portanto grava dois arquivos. Um único `baseline_classico.csv` tornaria impossível separar Naive Bayes de SVM na avaliação.

---

## Métricas e Avaliação

Todos os modelos são avaliados sobre o **mesmo conjunto de teste**, gerado pelo split fixo do notebook 01. A classe positiva é **smishing**.

### Matriz de confusão

Artefato central da avaliação, gerada para cada modelo:

```
                 Predito: Smishing   Predito: Legítimo
Real: Smishing        VP                  FN  ← erro mais caro
Real: Legítimo        FP                  VN
```

Implementação: `confusion_matrix`, `classification_report`, `ConfusionMatrixDisplay` do `sklearn`.

### Métricas reportadas

| Métrica | Fórmula | Papel |
|---|---|---|
| Acurácia | (VP + VN) / Total | Referência geral |
| Precisão | VP / (VP + FP) | Confiabilidade do alerta de golpe |
| Recall | VP / (VP + FN) | Quantos golpes o modelo efetivamente pega |
| F1-Score | 2·(P·R)/(P + R) | Balanço geral |
| **F2-Score** | 5·(P·R)/(4·P + R) | **Métrica-título** — pondera recall com peso 4x sobre precisão |
| Especificidade | VN / (VN + FP) | Acerto sobre mensagens legítimas |
| AUC-ROC | Área sob a curva ROC | Desempenho independente do limiar |

**Por que F2 como métrica-título.** Um falso negativo deixa o golpe chegar ao idoso; um falso positivo apenas sinaliza uma mensagem legítima. O custo é assimétrico, e o F2 formaliza essa assimetria com um número único. Priorizar recall isolado seria frágil — um classificador que responde "smishing" para tudo tem recall 1.0, e a banca apontaria isso na hora. O F2 mantém o recall no centro sem premiar o modelo degenerado.

**Sobre a acurácia:** permanece na tabela, conforme a 4.3.4. Comentar na discussão que, sob desbalanceamento, ela é otimista — mas reportar.

**Calibração do limiar de decisão.** Para os clássicos e o BERTimbau, o limiar não fica em 0.5 por omissão: escolhe-se na **validação** o limiar que maximiza F2, e aplica-se esse limiar ao teste. Reportar as métricas nos dois limiares (0.5 e calibrado) evidencia o ganho e transforma a priorização do recall em decisão de engenharia concreta — insumo direto para as diretrizes da 4.4.5.

**Teste de McNemar (opcional, recomendado).** Verifica se a diferença entre dois modelos é estatisticamente significativa. Com um teste de 15% de um corpus provavelmente pequeno, uma diferença de F1 de 0.03 pode ser ruído — e "o modelo X foi melhor que o Y" sem teste de significância é pergunta previsível de banca. Se forem comparados mais de dois modelos, aplicar correção para múltiplas comparações (Holm ou Bonferroni). Se entrar no código, entra também na seção 4.3.4 do texto.

---

## Decisões Arquiteturais

| Decisão | Escolha | Justificativa |
|---|---|---|
| Split | 70/15/15 estratificado, seed 42 | Reprodutibilidade e comparação justa |
| Split único vs. k-fold | Split único persistido | Garante exatamente o mesmo teste para todas as trilhas |
| Ordem augmentation/split | Split primeiro, sintéticas só no treino | Val e test permanecem 100% reais |
| GPT vs. Llama | Llama aberto quantizado | Gratuito, reprodutível, sem API paga |
| Naive Bayes | ComplementNB | Variante otimizada para classes desbalanceadas |
| Métrica-título | F2 | Formaliza o custo assimétrico do falso negativo |
| Limiar de decisão | Calibrado na validação por F2 | Evita comparar modelos em um ponto arbitrário |
| Texto de entrada | Normalizado (clássicos) / original (BERTimbau, Llama) | BERTimbau é cased; TF-IDF se beneficia da normalização |
| Runtime | CPU: 01, 02, 05 / GPU: 00, 03, 04 | Uso racional da cota gratuita do Colab |
| Código | Repositório Git | Versionamento e reprodutibilidade; evita cache de módulo do Drive |
| Dados e resultados | Google Drive | Persistência entre sessões do Colab |

---

## Mapeamento com a Metodologia do TCC

| Seção da monografia | Onde é implementada |
|---|---|
| 4.3.1 Base de dados textual | `01_dados.ipynb` |
| 4.3.2 Pré-processamento (NLTK, spaCy) | `src/preprocessing.py`, usado em `02` |
| 4.3.3 Modelos de Linguagem Natural | `02`, `03`, `04` |
| 4.3.4 Métricas de avaliação | `src/evaluation.py`, consolidado em `05` |
| 4.4.1 Levantamento de golpes virtuais | Taxonomia `tipo_golpe`, aplicada em `01` e analisada em `05` |
| 4.4.2 Coleta e construção da base | `00_augmentation.ipynb` + `01_dados.ipynb` |
| 4.4.3 Pré-processamento e treinamento | `02`, `03`, `04` |
| 4.4.4 Avaliação dos resultados | `05_avaliacao.ipynb` |
| 4.4.5 Diretrizes para soluções técnicas | Faixas de risco da rubrica + análise de erros de `05` |

> A coluna `tipo_golpe` é o que dá entrega concreta ao objetivo específico nº 1 ("Mapear os principais tipos de golpes virtuais direcionados a idosos no Brasil"). Sem ela, esse objetivo fica apenas no levantamento bibliográfico, sem contrapartida nos resultados.

---

## Ajustes necessários no texto do TCC

O documento atual é o **projeto de pesquisa do TCC I** ("Projeto de Pesquisa apresentado... como requisito parcial para obtenção de nota da disciplina Trabalho de Conclusão de Curso I", com cronograma de AGO/2025 a JAN/2026). Para o TCC II, além da capa, da folha de aprovação e do cronograma, os itens abaixo precisam de edição para que o texto e a implementação não divirjam.

### Obrigatórios — a implementação diverge do texto sem eles

| # | O que fazer | Onde | Por quê |
|---|---|---|---|
| 1 | Acrescentar **matriz de confusão**, **F2-Score** e **AUC-ROC**, com fórmula e justificativa | 4.3.4 | Vão ser reportados no notebook 05; hoje a seção declara apenas acurácia, precisão, recall e F1 |
| 2 | Escrever o argumento do **custo assimétrico do erro** (FN > FP no contexto de idosos) | 4.3.4 ou 4.4.4 | É o que fundamenta o F2 como métrica-título e a calibração do limiar |
| 3 | Remover **GPT** ou registrar a substituição por Llama | Resumo, Abstract, 3.3 | O resumo cita "BERTimbau, GPT e LLaMA"; se os resultados não têm GPT, é a primeira pergunta da banca |
| 4 | Descrever **LLaMA**, a **quantização 4-bit** e os **baselines clássicos** | 4.3.3 | A seção só tem uma frase genérica e a imagem do BERTimbau; NB e SVM aparecem apenas na 4.4.3 |
| 5 | Substituir "Foram utilizados **X** datasets" pelas fontes reais, com licença e papel de cada uma | 4.3.1 | Placeholder ainda no texto; definir também o papel do Smishtank |
| 6 | Registrar a **ordem augmentation/split** e a regra anti-vazamento | 4.4.2 | Decisão metodológica central; precisa estar declarada, não só implementada |
| 7 | Converter a seção 5 de expectativa para **resultados reais** | 5 | Está inteira no futuro ("pretende-se demonstrar", "espera-se") |

### Recomendados

| # | O que fazer | Onde |
|---|---|---|
| 8 | Escrever a seção **Arquitetura Transformer** (attention, self-attention, encoder/decoder, por que superaram RNN/LSTM) | 3.3.1 — hoje contém apenas o roteiro "Explique:" |
| 9 | Preencher as **fórmulas** das métricas (espaços em branco) e corrigir a numeração, hoje toda "4.3.4.1" | 4.3.4 |
| 10 | Resolver a **4.4.4 duplicada** e remover o bloco de anotações de trabalho ainda no corpo do texto | 4.4.4 |
| 11 | Registrar que a **rubrica devolve níveis de risco**, não apenas rótulo binário | 4.4.5 e seção 5 |
| 12 | Corrigir o **sumário**: faltam 4.1 a 4.4; a seção 5 aparece como "RESULTADOS ESPERADOS" e no corpo como "RESULTADOS" | Sumário |
| 13 | Acertar a **Lista de Ilustrações** (quase todas apontando para a p. 10) e o rótulo "Imagem 1", usado para duas figuras diferentes | Pré-textuais |
| 14 | Remover a marcação "(se puder, inserir aqui uma imagem)" | 3.1 |

### Referências e citações

- **Citadas no texto e ausentes da lista:** Da Silva et al. (2023), FEBRABAN, SPC Brasil, ITI (2023)
- **Divergências de ano:** Gil citado como 2008 e referenciado como 2007; Jurafsky citado como 2023 e como 2000; Kaspersky citado como 2023 e referenciado como 2025
- **Grafia:** "Bartot" na fonte do Diagrama 1 (correto: Bortot); "SpacCy" no título da 4.3.2.2 (correto: spaCy); "Neuramind" na fonte da Imagem 01 (correto: NeuralMind); "PNL" no título da 4.4.3 (correto: PLN)
- **A referir, se as fontes forem adotadas:** Timko & Rahman (2024) para o Smishtank; a fonte do corpus PT-BR, se houver

---

## Referências

- Almeida, T. A.; Gómez Hidalgo, J. M. (2011). *SMS Spam Collection*. UCI Machine Learning Repository.
- Bortot, E. N. et al. (2024). *Teias de engano: uma análise dos riscos e estratégias de prevenção aos golpes cibernéticos praticados contra pessoas idosas na era digital*. Contribuciones a Las Ciencias Sociales.
- Dimario, C.; Bacha, R.; Butka, B. (2024). *Combatting Senior Scams Using a Large Language Model-Created Rubric*. ACM.
- GOV.BR / CERT.br. *Golpe do Smishing: saiba como se proteger*. Fascículo.
- NeuralMind (2023). *BERTimbau: BERT pré-treinado para o português*. Disponível em: https://neuralmind.ai/bert/
- Timko, D.; Rahman, M. L. (2024). *Smishing Dataset I: Phishing SMS Dataset from Smishtank.com*. ACM CODASPY.
