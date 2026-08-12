"""
Métricas e avaliação comparativa — seções 4.3.4 e 4.4.4 da monografia.

Ponto único de cálculo de métricas do projeto: os notebooks 02, 03 e 04
usam as mesmas funções daqui que o notebook 05, o que garante que os
números da tabela final sejam os mesmos que cada trilha reportou.

Métricas declaradas na seção 4.3.4: acurácia, precisão, recall, F1.
Acréscimos aprovados: matriz de confusão, F2 e AUC-ROC — os três precisam
constar da 4.3.4 do texto, com fórmula, antes da entrega.

Convenção da matriz de confusão (classe positiva = smishing):

                     Predito: smishing   Predito: legítimo
    Real: smishing          VP                  FN   ← erro mais caro
    Real: legítimo          FP                  VN
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

import config as CFG

POS = CFG.CLASSE_POSITIVA
NEG = CFG.CLASSE_NEGATIVA

# Ordem das colunas do arquivo de predição. Todas as trilhas gravam isto.
COLUNAS_PREDICAO = ['id', 'rotulo_real', 'rotulo_predito', 'score', 'modelo']


# ── Métricas ───────────────────────────────────────────────────────────────

def matriz_confusao(y_real, y_pred):
    """Matriz de confusão com a classe positiva na primeira linha/coluna."""
    return confusion_matrix(y_real, y_pred, labels=[POS, NEG])


def calcular_metricas(y_real, y_pred, y_score=None):
    """Todas as métricas do projeto para um modelo.

    `y_score` é a pontuação da classe positiva; quando ausente, a AUC-ROC
    não é calculada e volta como None.
    """
    y_real = np.asarray(y_real)
    y_pred = np.asarray(y_pred)

    cm = matriz_confusao(y_real, y_pred)
    VP, FN = int(cm[0, 0]), int(cm[0, 1])
    FP, VN = int(cm[1, 0]), int(cm[1, 1])

    especificidade = VN / (VN + FP) if (VN + FP) else 0.0

    auc = None
    if y_score is not None:
        y_score = np.asarray(y_score, dtype=float)
        # roc_auc_score exige as duas classes presentes no vetor real
        if len(np.unique(y_real)) == 2 and not np.isnan(y_score).any():
            auc = float(roc_auc_score((y_real == POS).astype(int), y_score))

    return {
        'VP': VP, 'FN': FN, 'FP': FP, 'VN': VN,
        'acuracia':       float(accuracy_score(y_real, y_pred)),
        'precisao':       float(precision_score(y_real, y_pred, pos_label=POS, zero_division=0)),
        'recall':         float(recall_score(y_real, y_pred, pos_label=POS, zero_division=0)),
        'f1':             float(fbeta_score(y_real, y_pred, beta=1, pos_label=POS, zero_division=0)),
        'f2':             float(fbeta_score(y_real, y_pred, beta=2, pos_label=POS, zero_division=0)),
        'especificidade': float(especificidade),
        'auc_roc':        auc,
    }


NOMES_METRICAS = {
    'acuracia':       'Acurácia',
    'precisao':       'Precisão',
    'recall':         'Recall',
    'f1':             'F1-Score',
    'f2':             'F2-Score',
    'especificidade': 'Especificidade',
    'auc_roc':        'AUC-ROC',
}

# F2 primeiro: é a métrica-título. Pondera o recall com peso 4x sobre a
# precisão, formalizando o custo assimétrico do falso negativo — um golpe
# não detectado chega ao idoso, um falso positivo apenas sinaliza uma
# mensagem legítima.
ORDEM_METRICAS = ['f2', 'recall', 'precisao', 'f1', 'especificidade', 'acuracia', 'auc_roc']


# ── Calibração do limiar ───────────────────────────────────────────────────

def calibrar_limiar(y_val, score_val, beta=2):
    """Escolhe, NA VALIDAÇÃO, o limiar que maximiza o F-beta.

    Deixar o limiar em 0.5 é aceitar um ponto de operação arbitrário. Como
    o projeto declara o falso negativo como erro mais caro, o limiar é uma
    decisão de projeto — e escolhê-lo na validação (nunca no teste) é o que
    torna essa decisão metodologicamente defensável.

    Devolve (limiar, f_beta_na_validacao).
    """
    y_bin = (np.asarray(y_val) == POS).astype(int)
    score = np.asarray(score_val, dtype=float)

    precisao, recall, limiares = precision_recall_curve(y_bin, score)

    # precision_recall_curve devolve um ponto a mais que `limiares`
    # (precisão=1, recall=0), que não corresponde a limiar nenhum.
    precisao, recall = precisao[:-1], recall[:-1]

    b2 = beta ** 2
    denominador = b2 * precisao + recall
    with np.errstate(divide='ignore', invalid='ignore'):
        f_beta = np.where(denominador > 0,
                          (1 + b2) * precisao * recall / denominador,
                          0.0)

    if len(f_beta) == 0:
        return 0.5, 0.0

    melhor = int(np.argmax(f_beta))
    return float(limiares[melhor]), float(f_beta[melhor])


def aplicar_limiar(scores, limiar):
    """Converte pontuações em rótulos usando o limiar dado."""
    scores = np.asarray(scores, dtype=float)
    return np.where(scores >= limiar, POS, NEG)


# ── Predições padronizadas ─────────────────────────────────────────────────

def salvar_predicoes(ids, y_real, y_pred, scores, nome_modelo, pasta=None):
    """Grava as predições no formato padronizado do projeto.

    Um arquivo por MODELO, não por notebook: o notebook 02 treina dois
    modelos e portanto grava dois arquivos. Um arquivo único impediria
    separar Naive Bayes de SVM na avaliação.

    O texto não é duplicado aqui — o notebook 05 recupera do test.csv
    pelo `id`, o que evita inconsistência e torna o `id` chave de verdade.
    """
    pasta = pasta or CFG.PATHS['predictions']
    os.makedirs(pasta, exist_ok=True)

    df = pd.DataFrame({
        'id':             np.asarray(ids),
        'rotulo_real':    np.asarray(y_real),
        'rotulo_predito': np.asarray(y_pred),
        'score':          np.asarray(scores, dtype=float),
        'modelo':         nome_modelo,
    })[COLUNAS_PREDICAO]

    caminho = f'{pasta}/{nome_modelo}.csv'
    df.to_csv(caminho, index=False, encoding='utf-8')
    print(f'[ok] {caminho}  ({len(df)} linhas)')
    return df


def carregar_predicoes(pasta=None):
    """Lê todos os arquivos de predição. Devolve {nome_modelo: DataFrame}."""
    pasta = pasta or CFG.PATHS['predictions']

    predicoes = {}
    for arquivo in sorted(os.listdir(pasta)):
        if not arquivo.endswith('.csv') or 'checkpoint' in arquivo:
            continue
        df = pd.read_csv(f'{pasta}/{arquivo}', encoding='utf-8')

        faltando = [c for c in COLUNAS_PREDICAO if c not in df.columns]
        if faltando:
            print(f'[AVISO] {arquivo} ignorado — colunas ausentes: {faltando}')
            continue

        nome = df['modelo'].iloc[0] if len(df) else arquivo[:-4]
        predicoes[nome] = df
        print(f'[ok] {arquivo}  ({len(df)} linhas)  modelo={nome}')

    return predicoes


def verificar_integridade(predicoes):
    """Confirma que todos os modelos avaliaram exatamente os mesmos exemplos.

    Qualquer divergência aqui invalida a comparação — é o pressuposto que
    sustenta a afirmação de que as trilhas foram avaliadas em condições
    iguais. Devolve True se estiver tudo certo.
    """
    if not predicoes:
        print('[ERRO] Nenhuma predição carregada.')
        return False

    nomes = list(predicoes)
    referencia = nomes[0]
    ids_ref = set(predicoes[referencia]['id'])
    tudo_ok = True

    print('=== Integridade da comparação ===')
    for nome in nomes[1:]:
        ids = set(predicoes[nome]['id'])
        diff = ids_ref.symmetric_difference(ids)
        if diff:
            print(f'  [ERRO] {referencia} vs {nome}: {len(diff)} IDs divergentes')
            tudo_ok = False
        else:
            print(f'  [ok] {referencia} vs {nome}: mesmos {len(ids)} IDs')

    # Os rótulos reais também precisam bater — não basta o conjunto de IDs
    base = predicoes[referencia].set_index('id')['rotulo_real']
    for nome in nomes[1:]:
        outro = predicoes[nome].set_index('id')['rotulo_real']
        comum = base.index.intersection(outro.index)
        divergentes = (base.loc[comum] != outro.loc[comum]).sum()
        if divergentes:
            print(f'  [ERRO] {nome}: {divergentes} rótulos reais divergentes de {referencia}')
            tudo_ok = False

    if tudo_ok:
        n_pos = int((predicoes[referencia]['rotulo_real'] == POS).sum())
        n_neg = int((predicoes[referencia]['rotulo_real'] == NEG).sum())
        print(f'\n  Conjunto de teste: {len(ids_ref)} mensagens '
              f'({n_pos} {POS}, {n_neg} {NEG})')

    return tudo_ok


# ── Tabela comparativa ─────────────────────────────────────────────────────

def tabela_comparativa(predicoes, salvar=True):
    """Calcula as métricas de todos os modelos e monta a tabela final."""
    linhas = {}
    for nome, df in predicoes.items():
        score = df['score'] if 'score' in df.columns else None
        linhas[nome] = calcular_metricas(df['rotulo_real'], df['rotulo_predito'], score)

    tabela = pd.DataFrame(linhas).T
    ordem = ORDEM_METRICAS + ['VP', 'FN', 'FP', 'VN']
    tabela = tabela[[c for c in ordem if c in tabela.columns]]

    if salvar:
        os.makedirs(CFG.PATHS['metrics'], exist_ok=True)
        caminho = f"{CFG.PATHS['metrics']}/comparativo_final.csv"
        tabela.to_csv(caminho, encoding='utf-8')
        print(f'Tabela salva em: {caminho}')

    return tabela


def formatar_tabela(tabela):
    """Versão da tabela com nomes legíveis, para colar na monografia."""
    exibir = tabela[[c for c in ORDEM_METRICAS if c in tabela.columns]].copy()
    exibir = exibir.rename(columns=NOMES_METRICAS)
    return exibir.round(4)


# ── Figuras ────────────────────────────────────────────────────────────────

def plot_confusao(y_real, y_pred, titulo, ax=None, salvar_como=None):
    """Matriz de confusão de um modelo."""
    import matplotlib.pyplot as plt

    criou = ax is None
    if criou:
        _, ax = plt.subplots(figsize=(5, 4))

    cm = matriz_confusao(y_real, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=[POS, NEG]).plot(
        ax=ax, colorbar=False, cmap='Blues', values_format='d'
    )
    ax.set_title(titulo)
    ax.set_xlabel('Predito')
    ax.set_ylabel('Real')

    if salvar_como:
        _salvar_figura(plt, salvar_como)
    return ax


def plot_confusao_comparativo(predicoes, metricas=None, salvar_como='confusao_comparativo.png'):
    """Matrizes de confusão de todos os modelos lado a lado."""
    import matplotlib.pyplot as plt

    n = len(predicoes)
    if n == 0:
        print('[AVISO] Nada a plotar.')
        return None

    n_cols = 2 if n > 2 else n
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = np.atleast_1d(np.asarray(axes)).ravel()

    usados = 0
    for i, (nome, df) in enumerate(predicoes.items()):
        titulo = nome
        if metricas is not None and nome in metricas.index:
            m = metricas.loc[nome]
            titulo = f"{nome}\nF2={m['f2']:.3f}  Recall={m['recall']:.3f}"
        plot_confusao(df['rotulo_real'], df['rotulo_predito'], titulo, ax=axes[i])
        usados = i + 1

    for j in range(usados, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        'Matrizes de confusão — comparação entre modelos (conjunto de teste)\n'
        'Linha = rótulo real | Coluna = rótulo predito',
        fontsize=13,
    )
    plt.tight_layout()
    _salvar_figura(plt, salvar_como)
    return fig


def plot_roc(predicoes, metricas=None, salvar_como='roc_comparativo.png'):
    """Curvas ROC de todos os modelos que tenham pontuação."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))

    for nome, df in predicoes.items():
        if 'score' not in df.columns or df['score'].isna().all():
            print(f'[AVISO] {nome} sem pontuação — fora da curva ROC.')
            continue

        y_bin = (df['rotulo_real'] == POS).astype(int)
        if y_bin.nunique() < 2:
            continue

        fpr, tpr, _ = roc_curve(y_bin, df['score'])
        auc = None
        if metricas is not None and nome in metricas.index:
            auc = metricas.loc[nome, 'auc_roc']
        rotulo = f'{nome} (AUC={auc:.3f})' if auc is not None and not pd.isna(auc) else nome
        ax.plot(fpr, tpr, lw=2, label=rotulo)

    ax.plot([0, 1], [0, 1], '--', color='gray', lw=1, label='Aleatório (AUC=0.500)')
    ax.set_xlabel('Taxa de Falsos Positivos (1 – Especificidade)')
    ax.set_ylabel('Taxa de Verdadeiros Positivos (Recall)')
    ax.set_title('Curvas ROC — comparação entre modelos (conjunto de teste)')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc='lower right')

    plt.tight_layout()
    _salvar_figura(plt, salvar_como)
    return fig


def plot_barras(tabela, metricas=('f2', 'recall', 'precisao', 'f1', 'especificidade'),
                salvar_como='barras_comparativo.png'):
    """Comparativo de métricas em barras agrupadas."""
    import matplotlib.pyplot as plt

    metricas = [m for m in metricas if m in tabela.columns]
    modelos = list(tabela.index)
    x = np.arange(len(metricas))
    largura = 0.8 / max(len(modelos), 1)

    fig, ax = plt.subplots(figsize=(13, 5))

    for i, modelo in enumerate(modelos):
        valores = [tabela.loc[modelo, m] for m in metricas]
        deslocamento = (i - len(modelos) / 2 + 0.5) * largura
        barras = ax.bar(x + deslocamento, valores, largura, label=modelo, alpha=0.88)
        for barra, valor in zip(barras, valores):
            ax.text(barra.get_x() + barra.get_width() / 2, barra.get_height() + 0.01,
                    f'{valor:.2f}', ha='center', va='bottom', fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels([NOMES_METRICAS.get(m, m) for m in metricas])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel('Score')
    ax.set_title('Comparação de métricas entre modelos (conjunto de teste)')
    ax.legend(loc='upper right')
    ax.axhline(1.0, color='gray', lw=0.5, ls='--')

    plt.tight_layout()
    _salvar_figura(plt, salvar_como)
    return fig


def _salvar_figura(plt, nome):
    os.makedirs(CFG.PATHS['figures'], exist_ok=True)
    caminho = f"{CFG.PATHS['figures']}/{nome}"
    plt.savefig(caminho, dpi=150, bbox_inches='tight')
    print(f'Figura salva em: {caminho}')


# ── Significância estatística ──────────────────────────────────────────────

def mcnemar_par(df_a, df_b, nome_a='A', nome_b='B'):
    """Teste de McNemar entre dois modelos, alinhados por `id`.

    H0: os dois modelos erram nos mesmos casos. Só os pares discordantes
    (um acerta, o outro erra) carregam informação.
    """
    from statsmodels.stats.contingency_tables import mcnemar

    juntos = df_a[['id', 'rotulo_real', 'rotulo_predito']].merge(
        df_b[['id', 'rotulo_predito']].rename(columns={'rotulo_predito': 'pred_b'}),
        on='id',
    )

    acerto_a = juntos['rotulo_predito'] == juntos['rotulo_real']
    acerto_b = juntos['pred_b'] == juntos['rotulo_real']

    n11 = int((acerto_a & acerto_b).sum())      # ambos acertam
    n10 = int((acerto_a & ~acerto_b).sum())     # só A acerta
    n01 = int((~acerto_a & acerto_b).sum())     # só B acerta
    n00 = int((~acerto_a & ~acerto_b).sum())    # ambos erram

    discordantes = n10 + n01
    # Com poucos pares discordantes a aproximação qui-quadrado é ruim;
    # o teste exato (binomial) é o correto nesse regime.
    exato = discordantes < 25
    resultado = mcnemar([[n11, n10], [n01, n00]], exact=exato, correction=not exato)

    return {
        'modelo_a': nome_a, 'modelo_b': nome_b,
        'so_a_acerta': n10, 'so_b_acerta': n01,
        'discordantes': discordantes,
        'exato': exato,
        'p_valor': float(resultado.pvalue),
    }


def mcnemar_todos(predicoes, alfa=0.05):
    """McNemar em todos os pares, com correção de Holm.

    Comparar 4 modelos gera 6 testes; sem correção, a chance de um falso
    positivo entre eles passa de 25%. Holm é uniformemente mais poderoso
    que Bonferroni e igualmente simples de justificar no texto.
    """
    nomes = list(predicoes)
    resultados = [
        mcnemar_par(predicoes[nomes[i]], predicoes[nomes[j]], nomes[i], nomes[j])
        for i in range(len(nomes)) for j in range(i + 1, len(nomes))
    ]
    if not resultados:
        return pd.DataFrame()

    # Holm: ordena os p-valores e compara com alfa/(m - k)
    ordenados = sorted(range(len(resultados)), key=lambda k: resultados[k]['p_valor'])
    m = len(resultados)
    rejeitou_anterior = True
    for posicao, indice in enumerate(ordenados):
        limite = alfa / (m - posicao)
        significativo = rejeitou_anterior and resultados[indice]['p_valor'] <= limite
        resultados[indice]['alfa_holm'] = limite
        resultados[indice]['significativo'] = bool(significativo)
        rejeitou_anterior = significativo

    return pd.DataFrame(resultados)


# ── Análise de erros ───────────────────────────────────────────────────────

def analise_erros(predicoes, df_teste):
    """Junta as predições ao conjunto de teste para a análise qualitativa.

    Devolve um DataFrame com uma coluna por modelo, mais `n_erros` — quantos
    modelos erraram cada mensagem. Requer que `df_teste` traga `texto` e
    `tipo_golpe`, que é o que permite responder "qual tipo de golpe cada
    modelo mais erra?" (etapa 4.4.1 e parte qualitativa da seção 4.1).
    """
    colunas = ['id', 'texto', 'rotulo']
    if 'tipo_golpe' in df_teste.columns:
        colunas.append('tipo_golpe')
    base = df_teste[colunas].rename(columns={'rotulo': 'rotulo_real'}).copy()

    for nome, df in predicoes.items():
        base = base.merge(
            df[['id', 'rotulo_predito']].rename(columns={'rotulo_predito': nome}),
            on='id', how='left',
        )

    nomes = list(predicoes)
    base['n_erros'] = sum((base[n] != base['rotulo_real']).astype(int) for n in nomes)
    base['erraram_todos'] = base['n_erros'] == len(nomes)

    return base


def erros_por_tipo(analise, predicoes):
    """Taxa de erro de cada modelo por tipo de golpe."""
    if 'tipo_golpe' not in analise.columns:
        print('[AVISO] Coluna tipo_golpe ausente — inclua-a nos splits (notebook 01).')
        return pd.DataFrame()

    linhas = {}
    for nome in predicoes:
        erro = (analise[nome] != analise['rotulo_real'])
        linhas[nome] = erro.groupby(analise['tipo_golpe']).mean()

    return pd.DataFrame(linhas).round(4)


# ── Registro do split ──────────────────────────────────────────────────────

def salvar_split_info(splits, extras=None, caminho=None):
    """Grava o registro de proveniência do split (seed, proporções, contagens)."""
    caminho = caminho or CFG.SPLIT_INFO
    os.makedirs(os.path.dirname(caminho), exist_ok=True)

    info = {
        'seed': CFG.SEED,
        'proporcoes': CFG.SPLIT_RATIO,
        'classe_positiva': POS,
        'gerado_em': pd.Timestamp.now().isoformat(timespec='seconds'),
        'subconjuntos': {},
    }

    for nome, df in splits.items():
        contagem = df[CFG.COL_ROTULO].value_counts().to_dict()
        info['subconjuntos'][nome] = {
            'total': int(len(df)),
            'por_classe': {str(k): int(v) for k, v in contagem.items()},
            'prop_positiva': round(float((df[CFG.COL_ROTULO] == POS).mean()), 4),
            'sinteticas': int((df['fonte'] == 'sintetica').sum()) if 'fonte' in df.columns else 0,
        }

    if extras:
        info.update(extras)

    with open(caminho, 'w', encoding='utf-8') as arquivo:
        json.dump(info, arquivo, ensure_ascii=False, indent=2)

    print(f'Registro do split salvo em: {caminho}')
    return info
