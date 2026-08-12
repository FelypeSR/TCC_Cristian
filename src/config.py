"""
Configuração compartilhada por todos os notebooks.

Substitui o antigo par config.json + utils.py. Como o módulo vem do
repositório Git (e não do Drive), editar este arquivo e dar `git pull`
no notebook basta — não há o problema de módulo cacheado que acontece
quando o código é importado de dentro do Drive.

Uso:
    import sys
    sys.path.insert(0, '/content/tcc-smishing/src')
    import config as CFG

    CFG.fixar_seeds()
    df = pd.read_csv(CFG.SPLIT_FILES['train'])
"""

import os
import random

# ── Reprodutibilidade ──────────────────────────────────────────────────────

SEED = 42

# ── Raiz no Google Drive ───────────────────────────────────────────────────
# Único caminho a editar se a pasta do projeto mudar de lugar. Todos os
# demais são derivados dele em tempo de execução — não os escreva por
# extenso em lugar nenhum.

BASE = '/content/drive/MyDrive/TCC_Smishing'

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

SPLIT_FILES = {
    'train': f"{PATHS['splits']}/train.csv",
    'val':   f"{PATHS['splits']}/val.csv",
    'test':  f"{PATHS['splits']}/test.csv",
}

SPLIT_INFO = f"{PATHS['splits']}/split_info.json"

CORPUS_COMPLETO = f"{PATHS['processed']}/corpus_completo.csv"
SINTETICAS = f"{PATHS['synthetic']}/sinteticas_validadas.csv"

# ── Rótulos ────────────────────────────────────────────────────────────────
# A classe positiva é smishing. Nenhuma string de rótulo deve ser escrita
# solta no código dos notebooks: sempre CFG.CLASSE_POSITIVA / NEGATIVA.

CLASSE_POSITIVA = 'smishing'
CLASSE_NEGATIVA = 'legitima'
CLASSES = [CLASSE_POSITIVA, CLASSE_NEGATIVA]

# Índices numéricos para o BERTimbau. A classe positiva fica no índice 1
# por convenção — é o que `pos_label=1` do sklearn espera por padrão.
LABEL2ID = {CLASSE_NEGATIVA: 0, CLASSE_POSITIVA: 1}
ID2LABEL = {0: CLASSE_NEGATIVA, 1: CLASSE_POSITIVA}

COL_TEXTO = 'texto'
COL_ROTULO = 'rotulo'

# Mapeia rótulos das fontes externas para o vocabulário do projeto.
# Chaves sempre em minúsculas — a comparação é feita após .str.lower().
MAPA_ROTULOS = {
    # positivos
    'smishing': CLASSE_POSITIVA,
    'phishing': CLASSE_POSITIVA,
    'spam':     CLASSE_POSITIVA,
    'golpe':    CLASSE_POSITIVA,
    'fraude':   CLASSE_POSITIVA,
    'fraudulenta': CLASSE_POSITIVA,
    # negativos
    'legitima': CLASSE_NEGATIVA,
    'legítima': CLASSE_NEGATIVA,
    'ham':      CLASSE_NEGATIVA,
    'benign':   CLASSE_NEGATIVA,
    'normal':   CLASSE_NEGATIVA,
    'verdadeira': CLASSE_NEGATIVA,
}

# ── Taxonomia de golpes ────────────────────────────────────────────────────
# Atende ao objetivo específico nº 1 ("Mapear os principais tipos de golpes
# virtuais direcionados a idosos no Brasil") e à etapa 4.4.1. Derivada de
# Bortot et al. (2024) e das cartilhas do CERT.br / GOV.BR.
# Sem esta coluna no corpus, o objetivo 1 não tem contrapartida nos resultados.

TIPOS_GOLPE = [
    'banco',            # falso contato de instituição financeira
    'beneficio',        # INSS, aposentadoria, auxílio
    'premio',           # prêmio, sorteio, restituição
    'entrega',          # encomenda retida, taxa de liberação
    'emergencia',       # familiar em apuros, pedido urgente de dinheiro
    'orgao_publico',    # Receita, Detran, tribunal eleitoral
    'credito',          # empréstimo consignado, limite liberado
    'outro',
    'nao_aplicavel',    # mensagens legítimas
]

# ── Split ──────────────────────────────────────────────────────────────────

SPLIT_RATIO = {'train': 0.70, 'val': 0.15, 'test': 0.15}

# ── Modelos ────────────────────────────────────────────────────────────────

MODELOS = {
    'bertimbau': 'neuralmind/bert-base-portuguese-cased',
    'llama':     'meta-llama/Llama-3.2-3B-Instruct',
}

# ── Tokens especiais do pré-processamento ──────────────────────────────────
# Substituir URLs e telefones por tokens, em vez de removê-los, preserva a
# PRESENÇA dessas entidades como sinal — e presença de link é um dos
# indicadores mais fortes de smishing.

TOKENS = {
    'url':      'urltoken',
    'telefone': 'telefonetoken',
    'numero':   'numerotoken',
    'valor':    'valortoken',
    'email':    'emailtoken',
}

# ── Faixas de risco da rubrica (seção 4.4.5) ───────────────────────────────

FAIXAS_RISCO = [
    (0.00, 0.34, 'baixo'),
    (0.34, 0.67, 'medio'),
    (0.67, 1.01, 'alto'),
]


def nivel_risco(score):
    """Converte uma pontuação de risco contínua na faixa correspondente."""
    for minimo, maximo, nivel in FAIXAS_RISCO:
        if minimo <= score < maximo:
            return nivel
    return 'alto' if score >= 1.0 else 'baixo'


# ── Utilitários ────────────────────────────────────────────────────────────

def fixar_seeds(seed=SEED):
    """Fixa a seed em random, numpy, torch e transformers."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    try:
        import transformers
        transformers.set_seed(seed)
    except ImportError:
        pass

    return seed


def criar_pastas():
    """Cria a estrutura de pastas no Drive. Idempotente."""
    for nome, caminho in PATHS.items():
        os.makedirs(caminho, exist_ok=True)
    return PATHS


def resumo():
    """Imprime a configuração ativa — útil no topo de cada notebook."""
    print(f'seed            : {SEED}')
    print(f'base            : {BASE}')
    print(f'classe positiva : {CLASSE_POSITIVA}')
    print(f'bertimbau       : {MODELOS["bertimbau"]}')
    print(f'llama           : {MODELOS["llama"]}')
