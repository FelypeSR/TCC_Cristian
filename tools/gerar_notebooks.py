#!/usr/bin/env python3
"""Gera os notebooks do projeto tcc-smishing."""

import json
import os

DESTINO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notebooks")


def md(texto):
    return {"cell_type": "markdown", "metadata": {}, "source": texto.strip("\n").split("\n")}


def code(texto):
    linhas = texto.strip("\n").split("\n")
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": [l + "\n" for l in linhas[:-1]] + [linhas[-1]]}


def salvar(nome, celulas, gpu=False):
    meta = {
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": [], "toc_visible": True},
    }
    if gpu:
        meta["accelerator"] = "GPU"
        meta["colab"]["gpuType"] = "T4"

    nb = {"cells": celulas, "metadata": meta, "nbformat": 4, "nbformat_minor": 0}
    caminho = os.path.join(DESTINO, nome)
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"  {nome}  ({len(celulas)} células)")


SETUP = """
# ── Setup ──────────────────────────────────────────────────────────────────
# No Colab CADA notebook roda em um runtime próprio: instalar dependências
# em um notebook não vale para os outros. Por isso esta célula se repete em
# todos, e não existe um "notebook de instalação".

REPO = 'https://github.com/SEU-USUARIO/tcc-smishing.git'   # ← ajuste aqui

!git clone -q {REPO} /content/tcc-smishing 2>/dev/null || (cd /content/tcc-smishing && git pull -q)
!pip install -q -r /content/tcc-smishing/requirements.txt

from google.colab import drive
drive.mount('/content/drive')

import sys
sys.path.insert(0, '/content/tcc-smishing/src')

import config as CFG
CFG.fixar_seeds()
CFG.criar_pastas()
CFG.resumo()
"""

# ═══════════════════════════════════════════════════════════════════════════
# 00 — AUGMENTATION
# ═══════════════════════════════════════════════════════════════════════════

nb00 = [
md("""
# 00 — Geração de Mensagens Sintéticas

**TCC: Detecção de Smishing em Idosos com Modelos de Linguagem Natural**

> **Notebook de uso único.** Roda uma vez, produz um CSV versionado, e não faz
> parte do fluxo recorrente. Isso mantém o `01_dados` em CPU e evita disputar a
> cota de GPU com os notebooks 03 e 04.

**Respaldo na metodologia:** a seção 4.4.2 prevê "mensagens fraudulentas reais
(quando possível) **ou simuladas com base em evidências**", com rotulagem manual.
Este notebook implementa a parte de geração; a **validação manual é obrigatória**
e acontece na seção 5, antes de qualquer mensagem entrar no corpus.

### Regra que não pode ser quebrada

Cada mensagem gerada carrega o `id_semente` da mensagem real que a originou.
É esse campo que impede, no notebook 01, que variações da mesma semente caiam
em lados opostos do split — o vazamento mais provável e mais destrutivo deste
projeto.

### Saída

`data/synthetic/sinteticas_validadas.csv` — apenas as mensagens aprovadas na
revisão manual.
"""),
md("## 1. Setup"),
code(SETUP),
md("""
## 2. Carregamento das sementes

As sementes são as mensagens **reais** curadas a partir de Bortot et al. (2024)
e das cartilhas do CERT.br / GOV.BR. Precisam já estar padronizadas em
`data/raw/` com as colunas `texto`, `rotulo` e `tipo_golpe`.
"""),
code("""
import pandas as pd
import os

CAMINHO_SEMENTES = f"{CFG.PATHS['raw']}/sementes_pt.csv"

if not os.path.isfile(CAMINHO_SEMENTES):
    raise FileNotFoundError(
        f"Sementes não encontradas em {CAMINHO_SEMENTES}.\\n"
        "Faça o upload do CSV curado com as colunas: texto, rotulo, tipo_golpe"
    )

sementes = pd.read_csv(CAMINHO_SEMENTES, encoding='utf-8')

# Só mensagens de golpe são usadas como semente. Gerar mensagens "legítimas"
# sintéticas é arriscado: o modelo tende a produzir textos genéricos e
# artificiais, e o classificador aprenderia a separar estilo, não conteúdo.
sementes = sementes[sementes[CFG.COL_ROTULO] == CFG.CLASSE_POSITIVA].reset_index(drop=True)
sementes['id_semente'] = [f'semente_{i}' for i in range(len(sementes))]

print(f'Sementes de smishing: {len(sementes)}')
print(sementes['tipo_golpe'].value_counts().to_string())
sementes.head(3)
"""),
md("## 3. Carregamento do Llama quantizado"),
code("""
import torch
from google.colab import userdata
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Token nos Secrets do Colab (ícone da chave na barra lateral), nunca no código
HF_TOKEN = userdata.get('HF_TOKEN')
login(token=HF_TOKEN, add_to_git_credential=False)

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',          # NF4 é otimizado para pesos de redes neurais
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

MODEL_ID = CFG.MODELOS['llama']
tok = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
modelo = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb, device_map='auto', token=HF_TOKEN,
)
modelo.eval()

print(f'VRAM em uso: {torch.cuda.memory_allocated() / 1e9:.2f} GB')
"""),
md("""
## 4. Geração

Para cada semente, o modelo gera variações **do mesmo tipo de golpe**, mudando
a redação mas preservando o mecanismo de persuasão. Temperatura alta aqui é
proposital: queremos diversidade, não a resposta mais provável.
"""),
code("""
from tqdm.auto import tqdm

N_VARIACOES = 4      # por semente
TEMPERATURA = 0.9    # alta de propósito — o objetivo é variedade

INSTRUCAO = (
    'Você ajuda a construir um conjunto de dados acadêmico para TREINAR um '
    'detector de golpes por SMS que protege pessoas idosas no Brasil.\\n\\n'
    'A partir da mensagem de golpe abaixo, escreva {n} variações diferentes '
    'do MESMO tipo de golpe ({tipo}). Mude as palavras, a instituição citada '
    'e os valores, mas mantenha o mecanismo de persuasão.\\n\\n'
    'Regras:\\n'
    '- Cada variação em uma linha, numerada de 1 a {n}\\n'
    '- Máximo de 200 caracteres por mensagem\\n'
    '- Português do Brasil, no estilo real de SMS\\n'
    '- Não use dados de pessoas reais: sem nomes próprios, CPF ou contas verdadeiras\\n'
    '- Não escreva nada além das mensagens numeradas\\n\\n'
    'Mensagem original: {texto}'
)


def gerar_variacoes(texto, tipo, n=N_VARIACOES):
    mensagens = [{'role': 'user', 'content': INSTRUCAO.format(n=n, tipo=tipo, texto=texto)}]
    entrada = tok.apply_chat_template(
        mensagens, tokenize=True, add_generation_prompt=True, return_tensors='pt',
    ).to(modelo.device)

    with torch.no_grad():
        saida = modelo.generate(
            entrada, max_new_tokens=400, do_sample=True,
            temperature=TEMPERATURA, top_p=0.95,
            pad_token_id=tok.eos_token_id,
        )

    bruto = tok.decode(saida[0][entrada.shape[-1]:], skip_special_tokens=True)

    # Extrai as linhas numeradas, descartando o resto
    linhas = []
    for linha in bruto.split('\\n'):
        linha = linha.strip()
        if linha and linha[0].isdigit():
            texto_limpo = linha.lstrip('0123456789.)-  ').strip()
            if len(texto_limpo) > 20:
                linhas.append(texto_limpo)
    return linhas[:n]


geradas = []
for _, s in tqdm(sementes.iterrows(), total=len(sementes), desc='Gerando'):
    for variacao in gerar_variacoes(s[CFG.COL_TEXTO], s['tipo_golpe']):
        geradas.append({
            'texto': variacao,
            'rotulo': CFG.CLASSE_POSITIVA,
            'tipo_golpe': s['tipo_golpe'],
            'id_semente': s['id_semente'],   # ← rastreabilidade obrigatória
            'fonte': 'sintetica',
            'idioma': 'pt',
        })

df_geradas = pd.DataFrame(geradas)
print(f'\\nGeradas: {len(df_geradas)} mensagens a partir de {len(sementes)} sementes')
"""),
md("""
## 5. Deduplicação e preparação para a revisão manual

O modelo repete formulações com frequência. Deduplicamos antes de gastar tempo
de revisão humana com mensagens idênticas.
"""),
code("""
import preprocessing as pp

# Deduplica entre as geradas e contra as próprias sementes
df_geradas['_chave'] = df_geradas['texto'].apply(pp.chave_dedup)
chaves_sementes = set(sementes[CFG.COL_TEXTO].apply(pp.chave_dedup))

antes = len(df_geradas)
df_geradas = df_geradas.drop_duplicates(subset='_chave')
df_geradas = df_geradas[~df_geradas['_chave'].isin(chaves_sementes)]
df_geradas = df_geradas.drop(columns='_chave').reset_index(drop=True)

print(f'Duplicatas removidas: {antes - len(df_geradas)}')
print(f'Restantes para revisão: {len(df_geradas)}')

df_geradas['id'] = [f'sint_{i}' for i in range(len(df_geradas))]
df_geradas['aprovada'] = ''      # ← preencher na revisão manual: 1 = aprovada, 0 = descartada

CAMINHO_REVISAO = f"{CFG.PATHS['synthetic']}/sinteticas_para_revisao.csv"
df_geradas.to_csv(CAMINHO_REVISAO, index=False, encoding='utf-8')
print(f'\\nArquivo de revisão: {CAMINHO_REVISAO}')
"""),
md("""
## 6. ⚠️ Revisão manual — etapa obrigatória

A seção 4.4.2 da monografia exige rotulagem manual. Além disso, mensagens
geradas por LLM frequentemente saem implausíveis, fora do português brasileiro
coloquial, ou desalinhadas do tipo de golpe pedido.

**Faça agora, fora do notebook:**

1. Abra `data/synthetic/sinteticas_para_revisao.csv` no Planilhas Google
2. Para cada linha, preencha a coluna `aprovada`:
   - `1` — plausível como SMS real de golpe, coerente com o `tipo_golpe`
   - `0` — implausível, truncada, genérica demais ou fora do tipo
3. Corrija o `tipo_golpe` quando o modelo tiver desviado
4. Salve o arquivo **com o mesmo nome**

Registre no texto da monografia quantas foram geradas, quantas aprovadas e qual
o critério de descarte — a taxa de aprovação é resultado, não detalhe operacional.

Só depois disso execute a célula abaixo.
"""),
code("""
revisadas = pd.read_csv(CAMINHO_REVISAO, encoding='utf-8')

if revisadas['aprovada'].isna().all() or (revisadas['aprovada'].astype(str).str.strip() == '').all():
    raise ValueError(
        'A coluna "aprovada" está vazia — a revisão manual ainda não foi feita.\\n'
        'Ver instruções na célula acima.'
    )

aprovadas = revisadas[revisadas['aprovada'].astype(str).str.strip() == '1'].copy()
aprovadas = aprovadas.drop(columns=['aprovada'])

taxa = len(aprovadas) / len(revisadas) if len(revisadas) else 0
print(f'Geradas   : {len(revisadas)}')
print(f'Aprovadas : {len(aprovadas)}  ({taxa:.1%})')
print(f'Descartadas: {len(revisadas) - len(aprovadas)}')
print('\\nPor tipo de golpe:')
print(aprovadas['tipo_golpe'].value_counts().to_string())

aprovadas.to_csv(CFG.SINTETICAS, index=False, encoding='utf-8')
print(f'\\nSalvo em: {CFG.SINTETICAS}')
print('\\nProssiga para o notebook 01_dados.ipynb.')
"""),
]

# ═══════════════════════════════════════════════════════════════════════════
# 01 — DADOS
# ═══════════════════════════════════════════════════════════════════════════

nb01 = [
md("""
# 01 — Coleta, Rotulagem e Splits

**TCC: Detecção de Smishing em Idosos com Modelos de Linguagem Natural**

Constrói a base unificada e produz o **split único 70/15/15 (seed 42)** que
todos os modelos compartilham. A integridade desse split é o que sustenta a
afirmação de que as trilhas foram comparadas em condições iguais.

### A decisão mais importante do projeto

**Split primeiro, sintéticas depois — e só no treino.**

```
corpus real → split 70/15/15 estratificado
                    │
                    ├── train ← recebe as mensagens sintéticas
                    ├── val   ← 100% real
                    └── test  ← 100% real
```

Assim a monografia pode afirmar que *os modelos foram avaliados exclusivamente
sobre mensagens reais*. Isso responde de uma vez a duas perguntas de banca: a do
vazamento de dados e a de "seu modelo não está apenas reconhecendo o estilo do
gerador?".

Fazer o contrário — aumentar e depois dividir — coloca variações da mesma
mensagem-semente em treino e teste. Todas as métricas sobem, os três modelos
parecem ótimos, e o resultado não vale nada.

### Saídas

| Arquivo | Conteúdo |
|---|---|
| `data/processed/corpus_completo.csv` | `id, texto, rotulo, tipo_golpe, id_semente, fonte, idioma` |
| `data/splits/train.csv`, `val.csv`, `test.csv` | Split fixo, com todas as colunas acima |
| `data/splits/split_info.json` | Seed, proporções, contagens, nº de sintéticas |
"""),
md("## 1. Setup"),
code(SETUP),
code("""
import os
import pandas as pd
import numpy as np

import preprocessing as pp
import evaluation as ev

RAW = CFG.PATHS['raw']


def padronizar(df, fonte, idioma, tipo_padrao='outro'):
    \"\"\"Converte qualquer fonte para o esquema único do projeto.\"\"\"
    df = df.copy()
    df = df.dropna(subset=[CFG.COL_TEXTO])
    df = df[df[CFG.COL_TEXTO].astype(str).str.strip() != '']

    if 'tipo_golpe' not in df.columns:
        df['tipo_golpe'] = np.where(
            df[CFG.COL_ROTULO] == CFG.CLASSE_POSITIVA, tipo_padrao, 'nao_aplicavel')

    if 'id_semente' not in df.columns:
        df['id_semente'] = ''      # mensagens reais não têm semente

    df = df.reset_index(drop=True)
    df['id'] = [f'{fonte}_{i}' for i in range(len(df))]
    df['fonte'] = fonte
    df['idioma'] = idioma

    return df[['id', CFG.COL_TEXTO, CFG.COL_ROTULO, 'tipo_golpe',
               'id_semente', 'fonte', 'idioma']]


def mapear_rotulos(serie):
    \"\"\"Aplica o mapa do config, devolvendo NaN para rótulos desconhecidos.\"\"\"
    return serie.astype(str).str.strip().str.lower().map(CFG.MAPA_ROTULOS)


print('Utilitários definidos.')
"""),
md("""
## 2. SMS Spam Collection (UCI)

> **Papel: validação de pipeline apenas.** É spam genérico em inglês, não
> smishing em português. Serve para confirmar que a esteira de pré-processamento
> e modelagem funciona mecanicamente. **Não entra no corpus principal e não
> constitui achado do domínio do TCC** — no texto, esses resultados precisam
> aparecer claramente separados.
"""),
code("""
from ucimlrepo import fetch_ucirepo

sms = fetch_ucirepo(id=228)
df_sms = pd.DataFrame({
    CFG.COL_TEXTO: sms.data.features['sms'],
    # 'spam' aqui é mensagem indesejada; mapeada para a classe positiva apenas
    # para que o pipeline mecânico tenha duas classes com que trabalhar
    CFG.COL_ROTULO: sms.data.targets['label'].map(
        {'spam': CFG.CLASSE_POSITIVA, 'ham': CFG.CLASSE_NEGATIVA}),
})

df_sms = padronizar(df_sms, fonte='sms_spam_collection', idioma='en')
df_sms.to_csv(f'{RAW}/sms_spam_collection.csv', index=False, encoding='utf-8')

print(f'SMS Spam Collection: {len(df_sms)} mensagens')
print(df_sms[CFG.COL_ROTULO].value_counts().to_string())
"""),
md("""
## 3. Fontes curadas em PT-BR

Bortot et al. (2024) e CERT.br / GOV.BR **não são datasets prontos** — são
materiais curados que compõem o corpus principal. Faça o upload dos CSVs em
`data/raw/` antes de executar.

Colunas esperadas: `texto` (ou `mensagem`), `rotulo` e, idealmente, `tipo_golpe`.
"""),
code("""
# Ajuste os nomes de coluna conforme os arquivos reais
FONTES_PT = [
    {'arquivo': 'bortot.csv', 'fonte': 'bortot', 'col_texto': 'mensagem', 'col_rotulo': 'rotulo'},
    {'arquivo': 'certbr.csv', 'fonte': 'certbr', 'col_texto': 'mensagem', 'col_rotulo': 'rotulo'},
]


def carregar_fonte(spec):
    caminho = f"{RAW}/{spec['arquivo']}"
    if not os.path.isfile(caminho):
        print(f"[AVISO] {spec['arquivo']} não encontrado em data/raw/ — fonte ignorada.")
        return None

    bruto = pd.read_csv(caminho, encoding='utf-8')
    print(f"{spec['arquivo']}: colunas {bruto.columns.tolist()}")

    df = pd.DataFrame({
        CFG.COL_TEXTO:  bruto[spec['col_texto']],
        CFG.COL_ROTULO: mapear_rotulos(bruto[spec['col_rotulo']]),
    })
    if 'tipo_golpe' in bruto.columns:
        df['tipo_golpe'] = bruto['tipo_golpe']

    antes = len(df)
    df = df.dropna(subset=[CFG.COL_ROTULO])
    if len(df) < antes:
        print(f'  [INFO] {antes - len(df)} linhas descartadas por rótulo desconhecido')

    return padronizar(df, fonte=spec['fonte'], idioma='pt')


fontes_reais = [df for df in (carregar_fonte(s) for s in FONTES_PT) if df is not None]

if not fontes_reais:
    raise FileNotFoundError(
        'Nenhuma fonte PT-BR disponível. Faça o upload dos CSVs curados em data/raw/.'
    )

reais = pd.concat(fontes_reais, ignore_index=True)
print(f'\\nCorpus real PT-BR: {len(reais)} mensagens')
print(reais[CFG.COL_ROTULO].value_counts().to_string())
"""),
md("""
## 4. Deduplicação

Deduplicar por texto exato deixa passar variações triviais de espaçamento,
pontuação e caixa. Como as fontes se sobrepõem, a comparação é feita sobre a
forma normalizada.
"""),
code("""
reais['_chave'] = reais[CFG.COL_TEXTO].apply(pp.chave_dedup)

antes = len(reais)
reais = reais.drop_duplicates(subset='_chave').drop(columns='_chave').reset_index(drop=True)
print(f'Duplicatas removidas: {antes - len(reais)}')
print(f'Corpus real após deduplicação: {len(reais)}')

# IDs definitivos, atribuídos depois da deduplicação
reais['id'] = [f'real_{i}' for i in range(len(reais))]
"""),
md("""
## 5. Split estratificado — apenas mensagens reais

A estratificação preserva a proporção de classes nos três subconjuntos, o que é
essencial dado o desbalanceamento típico de corpora de spam/smishing.
"""),
code("""
from sklearn.model_selection import train_test_split

R = CFG.SPLIT_RATIO

temp, teste = train_test_split(
    reais, test_size=R['test'], random_state=CFG.SEED, stratify=reais[CFG.COL_ROTULO],
)
# proporção da validação relativa ao que sobrou: 0.15 / 0.85
treino, val = train_test_split(
    temp, test_size=R['val'] / (R['train'] + R['val']),
    random_state=CFG.SEED, stratify=temp[CFG.COL_ROTULO],
)

treino, val, teste = (d.reset_index(drop=True) for d in (treino, val, teste))

print('=== Split das mensagens reais ===')
for nome, d in [('treino', treino), ('val', val), ('teste', teste)]:
    print(f'  {nome:8}: {len(d):5}  ({len(d)/len(reais):.1%})  '
          f'{(d[CFG.COL_ROTULO] == CFG.CLASSE_POSITIVA).mean():.1%} smishing')
"""),
md("""
## 6. Incorporação das mensagens sintéticas — só no treino

Se o notebook 00 não foi executado, esta célula é ignorada e o corpus fica
apenas com mensagens reais.
"""),
code("""
if os.path.isfile(CFG.SINTETICAS):
    sinteticas = pd.read_csv(CFG.SINTETICAS, encoding='utf-8')
    sinteticas = padronizar(sinteticas, fonte='sintetica', idioma='pt')

    # Filtro 1 — uma variação só entra se a semente dela estiver no TREINO.
    # Se a semente caiu em val ou teste, a variação vaza informação do
    # conjunto de avaliação.
    sementes_treino = set(treino['id']) | set(treino['id_semente'])
    antes = len(sinteticas)
    sinteticas = sinteticas[
        sinteticas['id_semente'].isin(sementes_treino) | (sinteticas['id_semente'] == '')
    ]
    print(f'[INFO] {antes - len(sinteticas)} descartadas — semente fora do treino')

    # Filtro 2 — o modelo gerador pode ter produzido, por acaso, uma mensagem
    # quase idêntica a uma de val ou teste, mesmo partindo de semente do
    # treino. A comparação é sobre a forma normalizada, não sobre o texto
    # exato: variações de espaçamento e pontuação não podem servir de disfarce.
    chaves_avaliacao = (set(val[CFG.COL_TEXTO].apply(pp.chave_dedup)) |
                        set(teste[CFG.COL_TEXTO].apply(pp.chave_dedup)))
    antes = len(sinteticas)
    sinteticas = sinteticas[~sinteticas[CFG.COL_TEXTO].apply(pp.chave_dedup).isin(chaves_avaliacao)]
    print(f'[INFO] {antes - len(sinteticas)} descartadas — texto colide com val/teste')

    # Filtro 3 — redundância com o próprio treino: não agrega, só infla
    chaves_treino = set(treino[CFG.COL_TEXTO].apply(pp.chave_dedup))
    antes = len(sinteticas)
    sinteticas = sinteticas[~sinteticas[CFG.COL_TEXTO].apply(pp.chave_dedup).isin(chaves_treino)]
    print(f'[INFO] {antes - len(sinteticas)} descartadas — já presentes no treino')

    treino = pd.concat([treino, sinteticas], ignore_index=True)
    treino = treino.sample(frac=1, random_state=CFG.SEED).reset_index(drop=True)

    print(f'Sintéticas incorporadas ao treino: {len(sinteticas)}')
else:
    print('[INFO] Sem mensagens sintéticas — corpus apenas com mensagens reais.')

print(f'\\nTreino final: {len(treino)}  |  Val: {len(val)}  |  Teste: {len(teste)}')
"""),
md("""
## 7. Verificações de integridade

Qualquer falha aqui invalida todos os resultados a jusante. O notebook para.
"""),
code("""
falhas = []

# 7.1 — nenhum id em dois subconjuntos
ids = {'treino': set(treino['id']), 'val': set(val['id']), 'teste': set(teste['id'])}
for a, b in [('treino', 'val'), ('treino', 'teste'), ('val', 'teste')]:
    comum = ids[a] & ids[b]
    print(f'  [{"ok" if not comum else "ERRO"}] {a} ∩ {b}: {len(comum)} ids')
    if comum:
        falhas.append(f'{len(comum)} ids compartilhados entre {a} e {b}')

# 7.2 — nenhuma semente atravessando o split (o vazamento perigoso)
sem = {k: set(d.loc[d['id_semente'].astype(str) != '', 'id_semente'])
       for k, d in [('treino', treino), ('val', val), ('teste', teste)]}
for a, b in [('treino', 'val'), ('treino', 'teste'), ('val', 'teste')]:
    comum = sem[a] & sem[b]
    print(f'  [{"ok" if not comum else "ERRO"}] sementes {a} ∩ {b}: {len(comum)}')
    if comum:
        falhas.append(f'{len(comum)} sementes compartilhadas entre {a} e {b}')

# 7.3 — val e teste precisam ser 100% reais
for nome, d in [('val', val), ('teste', teste)]:
    n_sint = int((d['fonte'] == 'sintetica').sum())
    print(f'  [{"ok" if not n_sint else "ERRO"}] {nome} sem sintéticas: {n_sint} encontradas')
    if n_sint:
        falhas.append(f'{n_sint} sintéticas em {nome}')

# 7.4 — texto duplicado atravessando o split
chaves = {k: set(d[CFG.COL_TEXTO].apply(pp.chave_dedup))
          for k, d in [('treino', treino), ('val', val), ('teste', teste)]}
for a, b in [('treino', 'val'), ('treino', 'teste'), ('val', 'teste')]:
    comum = chaves[a] & chaves[b]
    print(f'  [{"ok" if not comum else "ERRO"}] textos {a} ∩ {b}: {len(comum)}')
    if comum:
        falhas.append(f'{len(comum)} textos idênticos entre {a} e {b}')

if falhas:
    raise AssertionError('Split inválido:\\n  - ' + '\\n  - '.join(falhas))
print('\\n[OK] Split íntegro.')
"""),
md("## 8. Salvamento"),
code("""
corpus = pd.concat([treino, val, teste], ignore_index=True)
corpus.to_csv(CFG.CORPUS_COMPLETO, index=False, encoding='utf-8')
print(f'Corpus completo: {CFG.CORPUS_COMPLETO}  ({len(corpus)} mensagens)')

splits = {'train': treino, 'val': val, 'test': teste}
for nome, d in splits.items():
    d.to_csv(CFG.SPLIT_FILES[nome], index=False, encoding='utf-8')
    print(f'  {nome}.csv  ({len(d)} linhas)')

ev.salvar_split_info(splits, extras={
    'fontes_reais': [s['fonte'] for s in FONTES_PT],
    'sinteticas_no_treino': int((treino['fonte'] == 'sintetica').sum()),
    'val_teste_apenas_reais': True,
})
"""),
md("## 9. Estatísticas e figuras"),
code("""
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Distribuição de classes por subconjunto
for ax, (nome, d) in zip(axes, splits.items()):
    contagem = d[CFG.COL_ROTULO].value_counts()
    barras = ax.bar(contagem.index, contagem.values, color=['#d62728', '#2ca02c'])
    ax.set_title(f'{nome} ({len(d)} msgs)')
    ax.set_ylabel('Quantidade')
    for barra in barras:
        ax.text(barra.get_x() + barra.get_width() / 2, barra.get_height(),
                str(int(barra.get_height())), ha='center', va='bottom', fontweight='bold')

plt.suptitle(f'Distribuição de classes por subconjunto (seed={CFG.SEED})')
plt.tight_layout()
plt.savefig(f"{CFG.PATHS['figures']}/01_split_classes.png", dpi=150, bbox_inches='tight')
plt.show()
"""),
code("""
# Tipos de golpe e comprimento das mensagens — insumo da etapa 4.4.1
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

golpes = corpus[corpus[CFG.COL_ROTULO] == CFG.CLASSE_POSITIVA]['tipo_golpe'].value_counts()
axes[0].barh(golpes.index, golpes.values, color='#4878CF')
axes[0].set_title('Mensagens por tipo de golpe')
axes[0].set_xlabel('Quantidade')

corpus['n_chars'] = corpus[CFG.COL_TEXTO].str.len()
for rotulo, cor in [(CFG.CLASSE_POSITIVA, '#d62728'), (CFG.CLASSE_NEGATIVA, '#2ca02c')]:
    axes[1].hist(corpus.loc[corpus[CFG.COL_ROTULO] == rotulo, 'n_chars'],
                 bins=40, alpha=0.6, label=rotulo, color=cor)
axes[1].set_title('Comprimento das mensagens por classe')
axes[1].set_xlabel('Número de caracteres')
axes[1].legend()

plt.tight_layout()
plt.savefig(f"{CFG.PATHS['figures']}/01_corpus_descritivo.png", dpi=150, bbox_inches='tight')
plt.show()

print(corpus.groupby(CFG.COL_ROTULO)['n_chars'].describe().round(1).to_string())
print('\\nProssiga para o notebook 02_baseline_classico.ipynb.')
"""),
]

# ═══════════════════════════════════════════════════════════════════════════
# 02 — BASELINE CLÁSSICO
# ═══════════════════════════════════════════════════════════════════════════

nb02 = [
md("""
# 02 — Baselines Clássicos (Naive Bayes e SVM)

**TCC: Detecção de Smishing em Idosos com Modelos de Linguagem Natural**

Estabelece o piso de desempenho, conforme a seção 4.4.3. Se BERTimbau e Llama
não superarem estes modelos de forma relevante, a complexidade adicional não se
justifica — e isso é um achado, não um fracasso.

| Modelo | Implementação | Justificativa |
|---|---|---|
| Complement Naive Bayes | `sklearn.naive_bayes.ComplementNB` | Variante do Naive Bayes otimizada para classes desbalanceadas |
| SVM linear | `LinearSVC` + `CalibratedClassifierCV` | Forte baseline para texto; a calibração fornece as pontuações |

**Pré-processamento:** NLTK e spaCy, conforme as seções 4.3.2.1 e 4.3.2.2, mais
substituição de URLs, telefones e valores por tokens especiais — o que preserva
a *presença* dessas entidades como sinal.

> O vectorizador TF-IDF é ajustado **somente no treino** e depois aplicado a val
> e teste. Ajustá-lo no corpus completo seria vazamento.
"""),
md("## 1. Setup"),
code(SETUP),
code("""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import preprocessing as pp
import evaluation as ev

# O modelo do spaCy precisa ser baixado a cada runtime do Colab
!python -m spacy download pt_core_news_sm -q

treino = pd.read_csv(CFG.SPLIT_FILES['train'], encoding='utf-8')
val    = pd.read_csv(CFG.SPLIT_FILES['val'],   encoding='utf-8')
teste  = pd.read_csv(CFG.SPLIT_FILES['test'],  encoding='utf-8')

print(f'Treino: {len(treino)}  |  Val: {len(val)}  |  Teste: {len(teste)}')
"""),
md("""
## 2. Pré-processamento

`preparar_classico` aplica normalização, substituição por tokens especiais e
lematização com spaCy. A remoção de stopwords fica **desligada por padrão**: SMS
têm 15 a 25 palavras, e marcadores de urgência e comando ("já", "agora", "você")
estão na lista de stopwords do NLTK — são exatamente o vocabulário do golpe.
A célula de ablação, ao final, compara as duas versões.
"""),
code("""
USAR_STOPWORDS = False   # ver ablação na seção 8

X_treino_txt = pp.preparar_classico(treino[CFG.COL_TEXTO], stopwords=USAR_STOPWORDS)
X_val_txt    = pp.preparar_classico(val[CFG.COL_TEXTO],    stopwords=USAR_STOPWORDS)
X_teste_txt  = pp.preparar_classico(teste[CFG.COL_TEXTO],  stopwords=USAR_STOPWORDS)

y_treino = treino[CFG.COL_ROTULO].values
y_val    = val[CFG.COL_ROTULO].values
y_teste  = teste[CFG.COL_ROTULO].values

print('=== Exemplo: original vs. processado ===')
for i in range(3):
    print(f'\\n  [{y_treino[i]}]')
    print(f'  orig: {treino[CFG.COL_TEXTO].iloc[i][:110]}')
    print(f'  proc: {X_treino_txt[i][:110]}')
"""),
md("""
## 3. Vetorização

TF-IDF com bigramas, mais as features artesanais (presença de URL, urgência,
instituição financeira, promessa de prêmio, proporção de maiúsculas). As
features artesanais são calculadas sobre o texto **original** — caixa alta e
pontos de exclamação se perdem na normalização.

> **Cada modelo recebe uma matriz diferente, de propósito.** O Naive Bayes é um
> modelo multinomial: ele pressupõe contagens/frequências de termos. Injetar
> features contínuas escaladas (`n_chars`, `prop_maiusculas`) viola essa
> premissa e, na prática, faz os pesos degenerarem — a lista de termos
> discriminativos vira ruído. O SVM é linear e não tem essa restrição, então
> aproveita as features artesanais normalmente.
"""),
code("""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack, csr_matrix

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),   # bigramas capturam padrões como "clique aqui"
    max_features=20_000,
    sublinear_tf=True,    # log(1 + tf) reduz o peso de termos muito frequentes
    min_df=2,             # ignora termos que aparecem em um único documento
)

# fit apenas no treino; val e teste só são transformados
T_treino = vectorizer.fit_transform(X_treino_txt)
T_val    = vectorizer.transform(X_val_txt)
T_teste  = vectorizer.transform(X_teste_txt)

# MinMaxScaler, não StandardScaler: valores centrados em zero seriam negativos
escala = MinMaxScaler()
F_treino = escala.fit_transform(pp.matriz_features(treino[CFG.COL_TEXTO]))
F_val    = escala.transform(pp.matriz_features(val[CFG.COL_TEXTO]))
F_teste  = escala.transform(pp.matriz_features(teste[CFG.COL_TEXTO]))

# Naive Bayes: só TF-IDF (premissa multinomial)
# SVM: TF-IDF + features artesanais
MATRIZES = {
    'naive_bayes': (T_treino, T_val, T_teste),
    'svm': (
        hstack([T_treino, csr_matrix(F_treino)]).tocsr(),
        hstack([T_val,    csr_matrix(F_val)]).tocsr(),
        hstack([T_teste,  csr_matrix(F_teste)]).tocsr(),
    ),
}

print(f'Vocabulário     : {len(vectorizer.vocabulary_)} termos')
print(f'Naive Bayes     : {MATRIZES["naive_bayes"][0].shape}  (só TF-IDF)')
print(f'SVM             : {MATRIZES["svm"][0].shape}  (TF-IDF + {F_treino.shape[1]} artesanais)')
"""),
md("## 4. Treinamento"),
code("""
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

# ComplementNB estima o complemento de cada classe — supera o MultinomialNB
# quando as classes são desbalanceadas, que é o caso aqui
cnb = ComplementNB(alpha=1.0)
cnb.fit(MATRIZES['naive_bayes'][0], y_treino)

# class_weight='balanced' compensa o desbalanceamento; sem isso o modelo
# tende a favorecer a classe majoritária, contrariando a priorização do recall.
# method='sigmoid': a calibração isotônica superajusta em corpus pequeno.
svm = CalibratedClassifierCV(
    LinearSVC(C=1.0, max_iter=5000, class_weight='balanced', random_state=CFG.SEED),
    cv=5, method='sigmoid',
)
svm.fit(MATRIZES['svm'][0], y_treino)

MODELOS = {'naive_bayes': cnb, 'svm': svm}
print('Modelos treinados.')
"""),
md("""
## 5. Calibração do limiar na validação

O limiar não fica em 0.5 por omissão. Como o falso negativo é o erro mais caro,
o ponto de operação é uma decisão de projeto — e escolhê-lo na **validação**
(nunca no teste) é o que a torna defensável.
"""),
code("""
limiares = {}

for nome, modelo in MODELOS.items():
    _, Xva, _ = MATRIZES[nome]
    idx_pos = list(modelo.classes_).index(CFG.CLASSE_POSITIVA)
    score_val = modelo.predict_proba(Xva)[:, idx_pos]

    limiar, f2_val = ev.calibrar_limiar(y_val, score_val, beta=2)
    limiares[nome] = limiar

    m05 = ev.calcular_metricas(y_val, ev.aplicar_limiar(score_val, 0.5), score_val)
    mca = ev.calcular_metricas(y_val, ev.aplicar_limiar(score_val, limiar), score_val)

    print(f'\\n{nome}  — limiar escolhido: {limiar:.4f}')
    print(f'  val @0.5       F2={m05["f2"]:.4f}  recall={m05["recall"]:.4f}  FN={m05["FN"]}')
    print(f'  val @calibrado F2={mca["f2"]:.4f}  recall={mca["recall"]:.4f}  FN={mca["FN"]}')
"""),
md("""
## 6. Avaliação no conjunto de teste

O teste é tocado **uma única vez**, aqui, com o limiar já fixado na validação.
"""),
code("""
resultados, predicoes_teste = {}, {}

for nome, modelo in MODELOS.items():
    _, _, Xte = MATRIZES[nome]
    idx_pos = list(modelo.classes_).index(CFG.CLASSE_POSITIVA)
    score = modelo.predict_proba(Xte)[:, idx_pos]
    pred = ev.aplicar_limiar(score, limiares[nome])

    resultados[nome] = ev.calcular_metricas(y_teste, pred, score)
    predicoes_teste[nome] = pred
    ev.salvar_predicoes(teste['id'], y_teste, pred, score, nome)

tabela = pd.DataFrame(resultados).T
print('\\n=== Teste ===')
print(ev.formatar_tabela(tabela).to_string())
"""),
code("""
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, (nome, metricas) in zip(axes, resultados.items()):
    ev.plot_confusao(y_teste, predicoes_teste[nome],
                     f'{nome}\\nF2={metricas["f2"]:.3f}  Recall={metricas["recall"]:.3f}', ax=ax)

plt.suptitle('Matrizes de confusão — baselines clássicos (conjunto de teste)')
plt.tight_layout()
plt.savefig(f"{CFG.PATHS['figures']}/02_confusao_baselines.png", dpi=150, bbox_inches='tight')
plt.show()
"""),
md("""
## 7. Termos mais discriminativos

Insumo da análise qualitativa (seção 4.1) e defesa contra a pergunta "o que o
modelo aprendeu?".

> **Sobre a leitura do `ComplementNB`:** o nome `feature_log_prob_` sugere pesos
> do complemento, o que levaria a inverter a ordenação. Não é o caso na
> implementação do sklearn: com o padrão `norm=False`, o atributo guarda o
> **negativo** do log das frequências do complemento, e a predição é feita por
> `argmax`. Peso **alto** significa, portanto, termo mais associado à classe —
> ordenação **decrescente**, igual ao `MultinomialNB`. A célula seguinte
> confirma isso empiricamente; não altere a ordenação sem rodar essa checagem.
"""),
code("""
# O ComplementNB foi treinado só com TF-IDF, então os nomes vêm apenas do
# vectorizer — é o que mantém esta análise legível.
nomes_features = np.array(vectorizer.get_feature_names_out())
N = 20

for i, classe in enumerate(cnb.classes_):
    # decrescente: no ComplementNB do sklearn (norm=False), peso ALTO = mais
    # associado à classe. Ver checagem na célula seguinte.
    top = np.argsort(cnb.feature_log_prob_[i])[::-1][:N]
    print(f'\\nTop {N} termos de [{classe.upper()}]:')
    print('  ' + ' | '.join(nomes_features[top]))
"""),
code("""
# Checagem da ordenação: em um corpus controlado, o ComplementNB deve produzir
# a mesma lista que o MultinomialNB, cuja leitura é inequívoca. Se estas duas
# listas divergirem, a ordenação acima está errada.
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

docs = ['golpe clique link banco urgente'] * 30 + ['bolo festa neta domingo receita'] * 30
rot = [CFG.CLASSE_POSITIVA] * 30 + [CFG.CLASSE_NEGATIVA] * 30

v_chk = CountVectorizer()
X_chk = v_chk.fit_transform(docs)
nomes_chk = np.array(v_chk.get_feature_names_out())

c_chk = ComplementNB().fit(X_chk, rot)
m_chk = MultinomialNB().fit(X_chk, rot)

for i, classe in enumerate(c_chk.classes_):
    top_c = set(nomes_chk[np.argsort(c_chk.feature_log_prob_[i])[::-1][:5]])
    top_m = set(nomes_chk[np.argsort(m_chk.feature_log_prob_[i])[::-1][:5]])
    print(f'  [{classe:9}] CNB == MNB: {"OK" if top_c == top_m else "DIVERGIU — inverta a ordenação"}')
"""),
md("""
## 8. Ablação: contribuição das mensagens sintéticas

Responde numericamente à pergunta "a augmentation ajudou?", transformando uma
escolha metodológica em resultado. Roda em CPU, custa minutos.
"""),
code("""
if 'fonte' in treino.columns and (treino['fonte'] == 'sintetica').any():
    so_reais = treino[treino['fonte'] != 'sintetica'].reset_index(drop=True)

    # Mesmo pipeline do Naive Bayes acima: só TF-IDF, refeito sobre o
    # treino reduzido (refazer o fit é obrigatório — o vocabulário muda)
    Xr_txt = pp.preparar_classico(so_reais[CFG.COL_TEXTO], stopwords=USAR_STOPWORDS)
    vec_r = TfidfVectorizer(ngram_range=(1, 2), max_features=20_000, sublinear_tf=True, min_df=2)
    Xr = vec_r.fit_transform(Xr_txt)
    Xte_r = vec_r.transform(X_teste_txt)

    linhas = {}
    for rotulo, (X_fit, y_fit, X_ev) in {
        'com sintéticas': (MATRIZES['naive_bayes'][0], y_treino, MATRIZES['naive_bayes'][2]),
        'só reais':       (Xr, so_reais[CFG.COL_ROTULO].values, Xte_r),
    }.items():
        modelo = ComplementNB(alpha=1.0).fit(X_fit, y_fit)
        idx = list(modelo.classes_).index(CFG.CLASSE_POSITIVA)
        s = modelo.predict_proba(X_ev)[:, idx]
        linhas[rotulo] = ev.calcular_metricas(y_teste, ev.aplicar_limiar(s, 0.5), s)

    print('=== Ablação (Naive Bayes, limiar 0.5) ===')
    print(ev.formatar_tabela(pd.DataFrame(linhas).T).to_string())
    print(f"\\nTreino com sintéticas: {len(treino)}  |  só reais: {len(so_reais)}")
else:
    print('[INFO] Sem mensagens sintéticas no treino — ablação não se aplica.')
"""),
md("""
## 9. Validação de pipeline (SMS Spam Collection)

Confirma que a esteira TF-IDF → modelo → métricas funciona mecanicamente.
**Este resultado não entra na comparação de modelos da monografia** e deve
aparecer no texto claramente separado dos resultados de domínio.
"""),
code("""
caminho_sms = f"{CFG.PATHS['raw']}/sms_spam_collection.csv"

if os.path.isfile(caminho_sms):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    sms = pd.read_csv(caminho_sms, encoding='utf-8')
    s_tr, s_te = train_test_split(sms, test_size=0.15, random_state=CFG.SEED,
                                  stratify=sms[CFG.COL_ROTULO])

    v = TfidfVectorizer(max_features=5000)
    m = ComplementNB().fit(v.fit_transform(s_tr[CFG.COL_TEXTO]), s_tr[CFG.COL_ROTULO])

    print('=== Validação de pipeline — SMS Spam Collection (inglês) ===')
    print('(resultado mecânico, NÃO representa o domínio do TCC)\\n')
    print(classification_report(s_te[CFG.COL_ROTULO],
                                m.predict(v.transform(s_te[CFG.COL_TEXTO])), zero_division=0))
else:
    print('[AVISO] sms_spam_collection.csv não encontrado — execute o notebook 01.')

print('\\nProssiga para o notebook 03_bertimbau.ipynb.')
"""),
]

# ═══════════════════════════════════════════════════════════════════════════
# 03 — BERTIMBAU
# ═══════════════════════════════════════════════════════════════════════════

nb03 = [
md("""
# 03 — Fine-tuning do BERTimbau

**TCC: Detecção de Smishing em Idosos com Modelos de Linguagem Natural**

**Modelo:** `neuralmind/bert-base-portuguese-cased` (BERTimbau — NeuralMind)

### Decisões de projeto

| Parâmetro | Valor | Justificativa |
|---|---|---|
| Texto de entrada | `texto` original | O BERTimbau é *cased*: maiúsculas carregam informação, e seu tokenizador de subpalavras já lida com a variação que a normalização do notebook 02 remove |
| `max_length` | 128 tokens | SMS raramente ultrapassa 160 caracteres |
| Batch size | 16 | Cabe na VRAM da T4 com margem |
| Épocas | 5 (máx.) | Com early stopping |
| Learning rate | 2e-5 | Padrão para fine-tuning de BERT |
| `fp16` | True | Precisão mista: reduz memória e acelera na T4 |
| Seleção do checkpoint | F2 na validação | Coerente com o custo assimétrico do erro |

> **Checkpoints ficam em `/content`, não no Drive.** Cada um tem ~440 MB;
> gravá-los via FUSE a cada época é lento e consome a cota de 15 GB. Só o modelo
> final é copiado para o Drive.
"""),
md("## 1. Setup"),
code(SETUP),
code("""
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

import evaluation as ev

print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "INDISPONÍVEL"}')
if torch.cuda.is_available():
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    raise RuntimeError('Troque o runtime para GPU: Ambiente de execução → Alterar tipo.')

treino = pd.read_csv(CFG.SPLIT_FILES['train'], encoding='utf-8')
val    = pd.read_csv(CFG.SPLIT_FILES['val'],   encoding='utf-8')
teste  = pd.read_csv(CFG.SPLIT_FILES['test'],  encoding='utf-8')

# Rótulos para índices: legitima=0, smishing=1
y_treino = treino[CFG.COL_ROTULO].map(CFG.LABEL2ID).values
y_val    = val[CFG.COL_ROTULO].map(CFG.LABEL2ID).values
y_teste  = teste[CFG.COL_ROTULO].map(CFG.LABEL2ID).values

print(f'\\nTreino: {len(treino)}  |  Val: {len(val)}  |  Teste: {len(teste)}')
print(f'Positivos no treino: {y_treino.sum()} / {len(y_treino)}')
"""),
md("## 2. Tokenização e dataset"),
code("""
from transformers import AutoTokenizer
from torch.utils.data import Dataset

MAX_LENGTH = 128
MODEL_ID = CFG.MODELOS['bertimbau']

tok = AutoTokenizer.from_pretrained(MODEL_ID)


def tokenizar(textos):
    return tok(list(textos), padding=True, truncation=True,
               max_length=MAX_LENGTH, return_tensors='pt')


class DatasetSmishing(Dataset):
    def __init__(self, encodings, rotulos):
        self.encodings = encodings
        self.rotulos = rotulos

    def __len__(self):
        return len(self.rotulos)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.rotulos[i], dtype=torch.long)
        return item


ds_treino = DatasetSmishing(tokenizar(treino[CFG.COL_TEXTO]), y_treino)
ds_val    = DatasetSmishing(tokenizar(val[CFG.COL_TEXTO]),    y_val)
ds_teste  = DatasetSmishing(tokenizar(teste[CFG.COL_TEXTO]),  y_teste)

print(f'Tokens do 1º exemplo: {tok.convert_ids_to_tokens(ds_treino[0]["input_ids"])[:18]} ...')
"""),
md("""
## 3. Modelo, métricas e perda ponderada

A perda é ponderada pelo inverso da frequência de cada classe. Sem isso o modelo
tende a favorecer a classe majoritária, o que contraria a priorização do recall
declarada no projeto.
"""),
code("""
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import EarlyStoppingCallback
from sklearn.metrics import fbeta_score, recall_score, precision_score
import torch.nn as nn

modelo = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID, num_labels=2, id2label=CFG.ID2LABEL, label2id=CFG.LABEL2ID,
)

# Pesos inversamente proporcionais à frequência
contagem = np.bincount(y_treino, minlength=2)
pesos = torch.tensor(len(y_treino) / (2 * contagem), dtype=torch.float32).to('cuda')
print(f'Contagem por classe: {contagem}  →  pesos: {pesos.tolist()}')

POS_ID = CFG.LABEL2ID[CFG.CLASSE_POSITIVA]


def compute_metrics(eval_pred):
    logits, rotulos = eval_pred
    pred = np.argmax(logits, axis=-1)
    return {
        'f2':        fbeta_score(rotulos, pred, beta=2, pos_label=POS_ID, zero_division=0),
        'f1':        fbeta_score(rotulos, pred, beta=1, pos_label=POS_ID, zero_division=0),
        'recall':    recall_score(rotulos, pred, pos_label=POS_ID, zero_division=0),
        'precisao':  precision_score(rotulos, pred, pos_label=POS_ID, zero_division=0),
    }


class TrainerPonderado(Trainer):
    \"\"\"Trainer com perda ponderada por classe.\"\"\"

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        rotulos = inputs.pop('labels')
        saidas = model(**inputs)
        perda = nn.CrossEntropyLoss(weight=pesos)(saidas.logits, rotulos)
        return (perda, saidas) if return_outputs else perda
"""),
md("## 4. Treinamento"),
code("""
# Checkpoints no disco LOCAL do Colab, não no Drive
CKPT = '/content/bertimbau_ckpt'

args = TrainingArguments(
    output_dir=CKPT,
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    eval_strategy='epoch',           # em transformers < 4.41 chama-se evaluation_strategy
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='f2',      # métrica-título do projeto
    greater_is_better=True,
    fp16=True,
    logging_steps=25,
    save_total_limit=2,
    seed=CFG.SEED,
    report_to='none',
)

trainer = TrainerPonderado(
    model=modelo, args=args,
    train_dataset=ds_treino, eval_dataset=ds_val,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# resume_from_checkpoint: se a sessão do Colab cair, reexecutar esta célula
# retoma de onde parou em vez de recomeçar do zero
retomar = os.path.isdir(CKPT) and any(d.startswith('checkpoint-') for d in os.listdir(CKPT))
print(f'Retomando de checkpoint: {retomar}')

resultado = trainer.train(resume_from_checkpoint=retomar)
print(f"\\nLoss final de treino: {resultado.metrics.get('train_loss', float('nan')):.4f}")
"""),
code("""
# Curvas de aprendizado
historico = trainer.state.log_history
logs_treino = [e for e in historico if 'loss' in e and 'eval_loss' not in e]
logs_val    = [e for e in historico if 'eval_loss' in e]

if logs_treino and logs_val:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot([e['step'] for e in logs_treino], [e['loss'] for e in logs_treino],
                 marker='.', label='Treino')
    axes[0].plot([e['step'] for e in logs_val], [e['eval_loss'] for e in logs_val],
                 marker='o', label='Validação')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Passo')
    axes[0].legend()

    epocas = [e['epoch'] for e in logs_val]
    for chave, estilo in [('eval_f2', '-o'), ('eval_recall', '--s'), ('eval_precisao', ':^')]:
        if chave in logs_val[0]:
            axes[1].plot(epocas, [e[chave] for e in logs_val], estilo, label=chave.replace('eval_', ''))
    axes[1].set_title('Métricas na validação')
    axes[1].set_xlabel('Época')
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    plt.suptitle('BERTimbau — curvas de aprendizado')
    plt.tight_layout()
    plt.savefig(f"{CFG.PATHS['figures']}/03_bert_curvas.png", dpi=150, bbox_inches='tight')
    plt.show()
"""),
md("## 5. Calibração do limiar na validação"),
code("""
def pontuacoes(dataset):
    \"\"\"Probabilidade da classe positiva para cada exemplo.\"\"\"
    logits = trainer.predict(dataset).predictions
    return torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, POS_ID]


score_val = pontuacoes(ds_val)
y_val_str = val[CFG.COL_ROTULO].values

limiar, f2_val = ev.calibrar_limiar(y_val_str, score_val, beta=2)

m05 = ev.calcular_metricas(y_val_str, ev.aplicar_limiar(score_val, 0.5), score_val)
mca = ev.calcular_metricas(y_val_str, ev.aplicar_limiar(score_val, limiar), score_val)

print(f'Limiar escolhido: {limiar:.4f}')
print(f'  val @0.5       F2={m05["f2"]:.4f}  recall={m05["recall"]:.4f}  FN={m05["FN"]}')
print(f'  val @calibrado F2={mca["f2"]:.4f}  recall={mca["recall"]:.4f}  FN={mca["FN"]}')
"""),
md("## 6. Avaliação no teste e salvamento"),
code("""
score_teste = pontuacoes(ds_teste)
y_teste_str = teste[CFG.COL_ROTULO].values
pred = ev.aplicar_limiar(score_teste, limiar)

metricas = ev.calcular_metricas(y_teste_str, pred, score_teste)
print('=== BERTimbau — teste ===')
for chave in ev.ORDEM_METRICAS:
    print(f'  {ev.NOMES_METRICAS[chave]:16}: {metricas[chave]:.4f}')
print(f"  VP={metricas['VP']}  FN={metricas['FN']}  FP={metricas['FP']}  VN={metricas['VN']}")

ev.salvar_predicoes(teste['id'], y_teste_str, pred, score_teste, 'bertimbau')

ev.plot_confusao(y_teste_str, pred,
                 f'BERTimbau\\nF2={metricas["f2"]:.3f}  Recall={metricas["recall"]:.3f}',
                 salvar_como='03_confusao_bertimbau.png')
plt.show()
"""),
code("""
# Só o modelo final vai para o Drive — os checkpoints ficam no disco local
DESTINO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notebooks")
trainer.save_model(DESTINO)
tok.save_pretrained(DESTINO)

import json
with open(f'{DESTINO}/limiar.json', 'w') as f:
    json.dump({'limiar': float(limiar), 'criterio': 'F2 na validação'}, f, indent=2)

print(f'Modelo salvo em: {DESTINO}')
print(os.listdir(DESTINO))
print('\\nProssiga para o notebook 04_llm_llama.ipynb.')
"""),
]

# ═══════════════════════════════════════════════════════════════════════════
# 04 — LLAMA
# ═══════════════════════════════════════════════════════════════════════════

nb04 = [
md("""
# 04 — Classificação via Rubrica com Llama

**TCC: Detecção de Smishing em Idosos com Modelos de Linguagem Natural**

Classificação por **rubrica**, no espírito de Dimario, Bacha e Butka (2024).

> **Nota metodológica (GPT → Llama):** o projeto originalmente previa GPT
> (OpenAI). Por restrições de custo e acessibilidade, optou-se por um modelo
> aberto quantizado, alinhado ao objetivo de avaliar soluções gratuitas e
> reprodutíveis: pesos públicos permitem replicação exata, o que uma API paga
> não garante. **O resumo, o abstract e a seção 3.3 da monografia ainda citam
> GPT e precisam ser atualizados.**

### Como a rubrica funciona

Em vez de pedir uma probabilidade — que um modelo de 3B calibra mal — pede-se
que o modelo marque **quais critérios de golpe estão presentes**. A pontuação é
calculada de forma determinística a partir dos critérios marcados. Isso dá
interpretabilidade (dá para dizer ao usuário *por que* a mensagem é suspeita,
o que alimenta as diretrizes da etapa 4.4.5), robustez e determinismo.

Os sete critérios estão ancorados em Bortot et al. (2024, p. 13), citado na
seção 3.2, e nos padrões descritos na seção 3.1 — ver `src/rubrica.py`.

### Anti-vazamento

Exemplos few-shot vêm **exclusivamente do treino**. A escolha do prompt é feita
na **validação** — iterar o prompt olhando o resultado do teste seria vazamento.
"""),
md("## 1. Setup"),
code(SETUP),
code("""
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

import rubrica as R
import evaluation as ev

print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "INDISPONÍVEL"}')
if not torch.cuda.is_available():
    raise RuntimeError('Troque o runtime para GPU: Ambiente de execução → Alterar tipo.')

treino = pd.read_csv(CFG.SPLIT_FILES['train'], encoding='utf-8')
val    = pd.read_csv(CFG.SPLIT_FILES['val'],   encoding='utf-8')
teste  = pd.read_csv(CFG.SPLIT_FILES['test'],  encoding='utf-8')

print(f'\\nTreino: {len(treino)} (fonte dos exemplos)  |  Val: {len(val)}  |  Teste: {len(teste)}')
display(R.tabela_criterios())
"""),
md("## 2. Carregamento do Llama quantizado em 4-bit"),
code("""
from google.colab import userdata
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

HF_TOKEN = userdata.get('HF_TOKEN')
login(token=HF_TOKEN, add_to_git_credential=False)

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

MODEL_ID = CFG.MODELOS['llama']
tok = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)

# padding à esquerda: obrigatório para geração em lote com modelos causais
tok.padding_side = 'left'
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

modelo = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb, device_map='auto', token=HF_TOKEN,
)
modelo.eval()

print(f'VRAM em uso: {torch.cuda.memory_allocated() / 1e9:.2f} GB')
"""),
md("""
## 3. Exemplos few-shot — anotação manual

Os exemplos precisam vir do **treino** e ter os critérios anotados à mão. São
poucos: seis mensagens. Anote com cuidado — eles definem o padrão que o modelo
vai imitar.

Execute a célula abaixo para ver os candidatos sorteados do treino, depois
preencha `ANOTACOES` na célula seguinte.
"""),
code("""
N_POR_CLASSE = 3

cand_pos = treino[treino[CFG.COL_ROTULO] == CFG.CLASSE_POSITIVA].sample(
    N_POR_CLASSE, random_state=CFG.SEED)
cand_neg = treino[treino[CFG.COL_ROTULO] == CFG.CLASSE_NEGATIVA].sample(
    N_POR_CLASSE, random_state=CFG.SEED)
candidatos = pd.concat([cand_pos, cand_neg]).reset_index(drop=True)

print('Anote os critérios de cada mensagem (ver tabela da seção 1):\\n')
for i, linha in candidatos.iterrows():
    print(f'[{i}] ({linha[CFG.COL_ROTULO]}) {linha[CFG.COL_TEXTO]}')
    print()
"""),
code("""
# ── Preencha: índice do candidato → lista de critérios presentes ───────────
# Mensagens legítimas normalmente ficam com lista vazia.
ANOTACOES = {
    0: [1, 2, 3],     # ← ajuste conforme as mensagens exibidas acima
    1: [1, 3, 6],
    2: [2, 4, 3],
    3: [],
    4: [],
    5: [],
}

exemplos = candidatos.copy()
exemplos['criterios'] = exemplos.index.map(ANOTACOES)
exemplos['resposta_esperada'] = exemplos['criterios'].apply(R.resposta_esperada)

# Intercala as classes para evitar viés de posição no prompt
exemplos = exemplos.sample(frac=1, random_state=CFG.SEED).reset_index(drop=True)

for _, e in exemplos.iterrows():
    print(f'  [{e["resposta_esperada"]:>12}] {e[CFG.COL_TEXTO][:80]}')

print('\\n=== Prompt montado (final) ===')
print(tok.apply_chat_template(
    R.construir_mensagens('Mensagem de exemplo', exemplos),
    tokenize=False, add_generation_prompt=True)[-700:])
"""),
md("""
## 4. Inferência em lote

Processar uma mensagem por vez, com o limite de 90 min de ociosidade e 12 h de
sessão do Colab, pode simplesmente não terminar. O lote resolve isso, e o
checkpoint protege contra queda de sessão.
"""),
code("""
from tqdm.auto import tqdm

BATCH = 8


def classificar_lote(textos, exemplos_fs):
    \"\"\"Roda a rubrica em um lote de mensagens.\"\"\"
    prompts = [
        tok.apply_chat_template(R.construir_mensagens(t, exemplos_fs),
                                tokenize=False, add_generation_prompt=True)
        for t in textos
    ]
    entrada = tok(prompts, return_tensors='pt', padding=True,
                  add_special_tokens=False).to(modelo.device)

    with torch.no_grad():
        saida = modelo.generate(
            **entrada,
            max_new_tokens=20,   # a resposta é uma lista curta de números
            do_sample=False,     # decodificação gulosa: determinística
            pad_token_id=tok.pad_token_id,
        )

    novos = saida[:, entrada['input_ids'].shape[-1]:]
    return [tok.decode(s, skip_special_tokens=True).strip() for s in novos]


def rodar(df, exemplos_fs, checkpoint=None, desc='Classificando'):
    \"\"\"Aplica a rubrica a um DataFrame inteiro, com checkpoint opcional.\"\"\"
    prontos = {}
    if checkpoint and os.path.isfile(checkpoint):
        anterior = pd.read_csv(checkpoint, encoding='utf-8')
        prontos = dict(zip(anterior['id'], anterior['resposta_bruta'].fillna('')))
        print(f'Checkpoint: {len(prontos)} mensagens já classificadas')

    pendentes = df[~df['id'].isin(prontos)].reset_index(drop=True)
    linhas = []

    for inicio in tqdm(range(0, len(pendentes), BATCH), desc=desc):
        pedaco = pendentes.iloc[inicio:inicio + BATCH]
        respostas = classificar_lote(pedaco[CFG.COL_TEXTO].tolist(), exemplos_fs)

        for (_, linha), bruta in zip(pedaco.iterrows(), respostas):
            prontos[linha['id']] = bruta

        if checkpoint and (inicio // BATCH) % 10 == 0:
            pd.DataFrame({'id': list(prontos), 'resposta_bruta': list(prontos.values())}) \\
              .to_csv(checkpoint, index=False, encoding='utf-8')

    if checkpoint:
        pd.DataFrame({'id': list(prontos), 'resposta_bruta': list(prontos.values())}) \\
          .to_csv(checkpoint, index=False, encoding='utf-8')

    for _, linha in df.iterrows():
        analise = R.classificar_resposta(prontos.get(linha['id'], ''))
        linhas.append({'id': linha['id'], **analise})

    return pd.DataFrame(linhas)


print('Funções de inferência definidas.')
"""),
md("""
## 5. Seleção do prompt na validação

Compara few-shot com zero-shot **na validação**. A versão escolhida aqui é a
única que toca o conjunto de teste — é isso que impede o vazamento por iteração
de prompt.
"""),
code("""
variantes = {
    'few-shot (6 exemplos)': exemplos,
    'zero-shot':             None,
}

escolhido, melhor_f2, limiar_escolhido = None, -1.0, 0.5

for nome, exemplos_fs in variantes.items():
    resultado = rodar(val, exemplos_fs, desc=f'val · {nome}')
    juntos = val[['id', CFG.COL_ROTULO]].merge(resultado, on='id')

    limiar, f2 = ev.calibrar_limiar(juntos[CFG.COL_ROTULO], juntos['score'], beta=2)
    invalidas = int((~juntos['resposta_valida']).sum())

    print(f'\\n{nome}')
    print(f'  F2 na validação  : {f2:.4f}  (limiar {limiar:.4f})')
    print(f'  respostas inválidas: {invalidas} / {len(juntos)} ({invalidas/len(juntos):.1%})')

    if f2 > melhor_f2:
        escolhido, melhor_f2, limiar_escolhido = exemplos_fs, f2, limiar

nome_escolhido = [k for k, v in variantes.items() if v is escolhido][0]
print(f'\\n→ Prompt escolhido: {nome_escolhido}  (F2={melhor_f2:.4f}, limiar={limiar_escolhido:.4f})')
"""),
md("## 6. Inferência no conjunto de teste"),
code("""
CHECKPOINT = f"{CFG.PATHS['predictions']}/llama_checkpoint.csv"

resultado = rodar(teste, escolhido, checkpoint=CHECKPOINT, desc='teste')
juntos = teste[['id', CFG.COL_ROTULO, 'tipo_golpe']].merge(resultado, on='id')

# Respostas fora do formato: contadas como abstenção e reportadas.
# Quando é preciso atribuir rótulo, atribui-se SMISHING — cair no rótulo
# negativo produziria justamente o erro mais caro do projeto.
invalidas = juntos[~juntos['resposta_valida']]
print(f'Respostas fora do formato: {len(invalidas)} / {len(juntos)} ({len(invalidas)/len(juntos):.1%})')
if len(invalidas):
    print('\\nExemplos:')
    for _, linha in invalidas.head(5).iterrows():
        print(f'  id={linha["id"]}  bruta={linha["resposta_bruta"]!r}')

pred = ev.aplicar_limiar(juntos['score'], limiar_escolhido)
pred = np.where(juntos['resposta_valida'], pred, CFG.CLASSE_POSITIVA)
"""),
code("""
y_real = juntos[CFG.COL_ROTULO].values
metricas = ev.calcular_metricas(y_real, pred, juntos['score'])

print('=== Llama (rubrica) — teste ===')
for chave in ev.ORDEM_METRICAS:
    valor = metricas[chave]
    print(f'  {ev.NOMES_METRICAS[chave]:16}: {valor:.4f}' if valor is not None
          else f'  {ev.NOMES_METRICAS[chave]:16}: –')
print(f"  VP={metricas['VP']}  FN={metricas['FN']}  FP={metricas['FP']}  VN={metricas['VN']}")

ev.salvar_predicoes(juntos['id'], y_real, pred, juntos['score'], 'llama')

ev.plot_confusao(y_real, pred,
                 f'Llama (rubrica)\\nF2={metricas["f2"]:.3f}  Recall={metricas["recall"]:.3f}',
                 salvar_como='04_confusao_llama.png')
plt.show()
"""),
md("""
## 7. Níveis de risco e explicações

A seção 5 da monografia promete uma rubrica capaz de atribuir **níveis de risco**
e servir como recurso educativo. Esta seção entrega as duas coisas, e é o insumo
direto das diretrizes da etapa 4.4.5.
"""),
code("""
juntos['nivel_risco'] = juntos['score'].apply(CFG.nivel_risco)

print('=== Distribuição dos níveis de risco no teste ===')
print(pd.crosstab(juntos['nivel_risco'], juntos[CFG.COL_ROTULO]).to_string())

print('\\n=== Critérios mais acionados nas mensagens de golpe ===')
golpes = juntos[juntos[CFG.COL_ROTULO] == CFG.CLASSE_POSITIVA]
contagem = pd.Series(
    [nome for lista in golpes['criterios_nomes'] for nome in lista]).value_counts()
print(contagem.to_string())

juntos.to_csv(f"{CFG.PATHS['predictions']}/llama_rubrica_detalhado.csv",
              index=False, encoding='utf-8')
"""),
code("""
# Exemplos de alerta como o usuário final o veria
print('=== Exemplos de alerta ao usuário (etapa 4.4.5) ===')
for _, linha in golpes.nlargest(3, 'score').iterrows():
    texto = teste.loc[teste['id'] == linha['id'], CFG.COL_TEXTO].iloc[0]
    print(f'\\nMensagem: {texto}')
    print(f'Risco: {linha["nivel_risco"].upper()} (pontuação {linha["score"]:.2f})')
    print(R.explicar(linha['criterios'], linha['nivel_risco']))
    print('-' * 70)

print('\\nProssiga para o notebook 05_avaliacao.ipynb.')
"""),
]

# ═══════════════════════════════════════════════════════════════════════════
# 05 — AVALIAÇÃO
# ═══════════════════════════════════════════════════════════════════════════

nb05 = [
md("""
# 05 — Avaliação Comparativa

**TCC: Detecção de Smishing em Idosos com Modelos de Linguagem Natural**

Compara as três trilhas sobre o **mesmo conjunto de teste** e produz os
artefatos finais da seção 4.4.4. É o único notebook que enxerga os resultados de
todos os modelos ao mesmo tempo.

### Métricas

| Métrica | Origem |
|---|---|
| Acurácia, Precisão, Recall, F1 | Declaradas na seção 4.3.4 |
| Matriz de confusão, **F2**, AUC-ROC | Acréscimos — precisam constar da 4.3.4 |

**F2 é a métrica-título.** Um falso negativo deixa o golpe chegar ao idoso; um
falso positivo apenas sinaliza uma mensagem legítima. O F2 pondera o recall com
peso 4x sobre a precisão, formalizando essa assimetria em um número único.
Priorizar o recall isolado seria frágil: um classificador que responde "smishing"
para tudo teria recall 1.0.
"""),
md("## 1. Setup"),
code(SETUP),
code("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import evaluation as ev

teste = pd.read_csv(CFG.SPLIT_FILES['test'], encoding='utf-8')
predicoes = ev.carregar_predicoes()
"""),
md("""
## 2. Verificação de integridade

Todos os modelos precisam ter avaliado exatamente os mesmos exemplos. Qualquer
divergência aqui invalida a comparação — é o pressuposto que sustenta a
afirmação de que as trilhas foram avaliadas em condições iguais.
"""),
code("""
if not ev.verificar_integridade(predicoes):
    raise AssertionError(
        'Os modelos não avaliaram o mesmo conjunto. Reexecute os notebooks '
        'de modelo sobre o split atual antes de comparar.'
    )
"""),
md("## 3. Tabela comparativa"),
code("""
tabela = ev.tabela_comparativa(predicoes)
exibir = ev.formatar_tabela(tabela)

print('=== Comparação sobre o conjunto de teste ===')
print(exibir.to_string())

# Destaca o melhor valor de cada métrica
display(exibir.style.highlight_max(axis=0, props='font-weight:bold; color:#c0392b'))
"""),
code("""
print('=== Matriz de confusão por modelo ===\\n')
for nome, df in predicoes.items():
    m = tabela.loc[nome]
    print(f'{nome}:  VP={int(m["VP"]):4}  FN={int(m["FN"]):4}  '
          f'FP={int(m["FP"]):4}  VN={int(m["VN"]):4}')
print('\\nFN = golpe que passou (erro mais caro)  |  FP = legítima sinalizada')
"""),
md("## 4. Figuras"),
code("""
ev.plot_confusao_comparativo(predicoes, tabela, salvar_como='05_confusao_comparativo.png')
plt.show()
"""),
code("""
ev.plot_roc(predicoes, tabela, salvar_como='05_roc_comparativo.png')
plt.show()
"""),
code("""
ev.plot_barras(tabela, salvar_como='05_barras_comparativo.png')
plt.show()
"""),
md("""
## 5. Significância estatística

Com um conjunto de teste de 15% de um corpus pequeno, uma diferença de F1 de
0.03 pode ser ruído. O teste de McNemar compara os **pares discordantes** — os
casos em que um modelo acerta e o outro erra.

Comparar quatro modelos gera seis testes; sem correção, a chance de ao menos um
falso positivo passa de 25%. Por isso a correção de Holm.
"""),
code("""
mcnemar = ev.mcnemar_todos(predicoes, alfa=0.05)

if len(mcnemar):
    print('=== Teste de McNemar (correção de Holm, α=0.05) ===')
    print(mcnemar[['modelo_a', 'modelo_b', 'so_a_acerta', 'so_b_acerta',
                   'p_valor', 'alfa_holm', 'significativo']].to_string(index=False))

    mcnemar.to_csv(f"{CFG.PATHS['metrics']}/mcnemar.csv", index=False, encoding='utf-8')

    n_sig = int(mcnemar['significativo'].sum())
    print(f'\\nPares com diferença significativa: {n_sig} de {len(mcnemar)}')
    if n_sig == 0:
        print('Nenhuma diferença resistiu à correção — reporte isso no texto. '
              'É um resultado legítimo e mostra rigor.')
"""),
md("""
## 6. Análise qualitativa dos erros

Atende à parte qualitativa declarada na seção 4.1 ("análise de características
linguísticas das mensagens fraudulentas") e ao que a seção 5 promete.
"""),
code("""
analise = ev.analise_erros(predicoes, teste)
nomes = list(predicoes)

fn_unanimes = analise[(analise['rotulo_real'] == CFG.CLASSE_POSITIVA) & analise['erraram_todos']]
fp_unanimes = analise[(analise['rotulo_real'] == CFG.CLASSE_NEGATIVA) & analise['erraram_todos']]

print(f'Falsos negativos unânimes (golpe que escapou de TODOS): {len(fn_unanimes)}')
print(f'Falsos positivos unânimes (legítima sinalizada por TODOS): {len(fp_unanimes)}')

pd.set_option('display.max_colwidth', 130)

if len(fn_unanimes):
    print('\\n=== Golpes que escaparam de todos os modelos ===')
    print('(o material mais rico da discussão — analise o que têm em comum)\\n')
    colunas = ['id', 'texto'] + (['tipo_golpe'] if 'tipo_golpe' in analise.columns else [])
    display(fn_unanimes[colunas].head(10))
"""),
code("""
# Taxa de erro por tipo de golpe — conecta com a etapa 4.4.1
por_tipo = ev.erros_por_tipo(analise, predicoes)
if len(por_tipo):
    print('=== Taxa de erro por tipo de golpe ===')
    print(por_tipo.to_string())

    fig, ax = plt.subplots(figsize=(11, 5))
    por_tipo.plot(kind='bar', ax=ax, width=0.8)
    ax.set_ylabel('Taxa de erro')
    ax.set_xlabel('Tipo de golpe')
    ax.set_title('Taxa de erro por tipo de golpe e por modelo')
    ax.legend(title='Modelo')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(f"{CFG.PATHS['figures']}/05_erros_por_tipo.png", dpi=150, bbox_inches='tight')
    plt.show()

    por_tipo.to_csv(f"{CFG.PATHS['metrics']}/erros_por_tipo.csv", encoding='utf-8')
"""),
code("""
# Distribuição do número de modelos que erram cada mensagem
fig, ax = plt.subplots(figsize=(7, 4))
contagem = analise['n_erros'].value_counts().sort_index()
barras = ax.bar(contagem.index, contagem.values, color='#4878CF', edgecolor='white')
ax.set_xlabel('Número de modelos que erraram')
ax.set_ylabel('Número de mensagens')
ax.set_title('Distribuição de erros por mensagem')
ax.set_xticks(range(len(nomes) + 1))
for barra in barras:
    ax.text(barra.get_x() + barra.get_width() / 2, barra.get_height(),
            str(int(barra.get_height())), ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig(f"{CFG.PATHS['figures']}/05_distribuicao_erros.png", dpi=150, bbox_inches='tight')
plt.show()
"""),
md("## 7. Resumo final"),
code("""
melhor_f2     = tabela['f2'].idxmax()
melhor_recall = tabela['recall'].idxmax()
melhor_auc    = tabela['auc_roc'].idxmax() if tabela['auc_roc'].notna().any() else '–'

print('=' * 62)
print('RESUMO — CONJUNTO DE TESTE')
print('=' * 62)
print(f'  Melhor F2 (métrica-título) : {melhor_f2}')
print(f'  Melhor Recall (smishing)   : {melhor_recall}')
print(f'  Melhor AUC-ROC             : {melhor_auc}')
print()
print(exibir.to_string())
print()
print('O F2 é a métrica-título: pondera o recall com peso 4x sobre a precisão,')
print('formalizando o custo assimétrico do erro. Um falso negativo deixa o golpe')
print('chegar ao idoso; um falso positivo apenas sinaliza uma mensagem legítima.')
print('=' * 62)

print('\\n=== Arquivos gerados ===')
import os
for pasta in ['metrics', 'figures']:
    print(f'\\n{pasta}/')
    for arquivo in sorted(os.listdir(CFG.PATHS[pasta])):
        print(f'  {arquivo}')
"""),
]

if __name__ == "__main__":
    os.makedirs(DESTINO, exist_ok=True)
    print("Gerando notebooks:")
    salvar("00_augmentation.ipynb", nb00, gpu=True)
    salvar("01_dados.ipynb", nb01)
    salvar("02_baseline_classico.ipynb", nb02)
    salvar("03_bertimbau.ipynb", nb03, gpu=True)
    salvar("04_llm_llama.ipynb", nb04, gpu=True)
    salvar("05_avaliacao.ipynb", nb05)
    print("OK")
