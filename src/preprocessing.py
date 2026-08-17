"""
Pré-processamento textual — seções 4.3.2 e 4.4.3 da monografia.

Duas camadas, com propósitos distintos:

  1. `normalizar()` — limpeza e substituição por tokens especiais.
     Alimenta o TF-IDF dos modelos clássicos (notebook 02).

  2. `preparar_classico()` — normalização + tokenização (NLTK) +
     lematização (spaCy), conforme os instrumentos declarados nas
     seções 4.3.2.1 e 4.3.2.2.

O BERTimbau e o Llama NÃO usam nada disto: recebem o texto original.
O BERTimbau é *cased* — maiúsculas carregam informação — e seu tokenizador
de subpalavras já lida com a variação de superfície que a normalização
remove. Essa diferença é decisão fundamentada, não descuido, e precisa
aparecer no texto da monografia.

Ordem das substituições importa: e-mail e URL antes de telefone, telefone
antes de número. Caso contrário o padrão de número consome os dígitos que
o padrão de telefone precisaria ver.
"""

import hashlib
import re
import unicodedata

import config as CFG

TOKENS = CFG.TOKENS

# ── Padrões ────────────────────────────────────────────────────────────────

_CONTROLE = re.compile(
    r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f​‌‍⁠﻿]'
)

_EMAIL = re.compile(r'\b[\w.+-]+@[\w-]+\.[\w.-]+\b', re.IGNORECASE)

_URL = re.compile(
    # com protocolo ou www
    r'(?:https?://|www\.)\S+'
    # domínio com TLD conhecido, com ou sem caminho
    r'|\b[\w-]+(?:\.[\w-]+)*\.(?:com|com\.br|net|org|br|info|xyz|top|link|'
    r'site|online|shop|app|club|vip|win|store|fun)(?:/\S*)?\b'
    # qualquer domínio.tld SEGUIDO DE CAMINHO — pega encurtadores
    # (bit.ly/x, t.co/x, cutt.ly/x, is.gd/x), que são o padrão mais comum
    # em smishing e escapavam da lista fixa de TLDs acima
    r'|\b[\w-]+\.[a-z]{2,4}/\S+',
    re.IGNORECASE,
)

# Telefone: exige DDD OU um separador explícito no meio. Sem isso, o padrão
# antigo casava qualquer sequência de 8 dígitos — datas, códigos de rastreio,
# números de protocolo — e os rotulava como telefone.
_TELEFONE = re.compile(
    r'(?<!\d)'
    r'(?:'
    r'(?:\+?55[\s.\-]?)?\(?\d{2}\)?[\s.\-]?9?\d{4}[\s.\-]?\d{4}'  # com DDD
    r'|'
    r'9?\d{4}[\s.\-]\d{4}'                                        # sem DDD, com separador
    r')'
    r'(?!\d)'
)

# Valores monetários antes dos números soltos: "R$ 1.500,00" é um sinal
# diferente de um número qualquer, e é frequente em golpes de crédito e prêmio.
_VALOR = re.compile(r'R\$\s?\d[\d.,]*', re.IGNORECASE)

_NUMERO = re.compile(r'(?<!\w)\d+(?:[.,]\d+)*(?!\w)')

_PONTUACAO = re.compile(r"[^\w\s'\-]", re.UNICODE)
_QUEBRAS = re.compile(r'[\t\r\n]+')
_ESPACOS = re.compile(r' {2,}')


# ── Normalização ───────────────────────────────────────────────────────────

def normalizar_unicode(texto):
    """Forma NFC e remoção de caracteres de controle e zero-width.

    NFC mantém as letras acentuadas compostas (ao contrário de NFD, que as
    decompõe em base + acento e quebraria a contagem de caracteres). Os
    zero-width são comuns em mensagens copiadas de aplicativos e podem ser
    usados deliberadamente para escapar de filtros por palavra-chave.
    """
    texto = unicodedata.normalize('NFC', texto)
    return _CONTROLE.sub('', texto)


def substituir_entidades(texto):
    """Troca e-mails, URLs, telefones, valores e números pelos tokens.

    Aplicado ANTES do lowercase para não perder padrões em maiúsculas
    (por exemplo 'HTTP://' ou 'R$').
    """
    texto = _EMAIL.sub(TOKENS['email'], texto)
    texto = _URL.sub(TOKENS['url'], texto)
    texto = _TELEFONE.sub(TOKENS['telefone'], texto)
    texto = _VALOR.sub(TOKENS['valor'], texto)
    texto = _NUMERO.sub(TOKENS['numero'], texto)
    return texto


def normalizar_espacos(texto):
    """Quebras de linha e tabs viram espaço; espaços múltiplos, um só."""
    texto = _QUEBRAS.sub(' ', texto)
    texto = _ESPACOS.sub(' ', texto)
    return texto.strip()


def remover_pontuacao(texto):
    """Remove pontuação, preservando apóstrofo e hífen.

    Substitui por espaço em vez de string vazia: 'banco/caixa' deve virar
    dois tokens, não um único 'bancocaixa'.
    """
    return _PONTUACAO.sub(' ', texto)


def normalizar(texto):
    """Pipeline de normalização. Entrada do TF-IDF dos modelos clássicos."""
    if not isinstance(texto, str):
        return ''
    texto = normalizar_unicode(texto)
    texto = substituir_entidades(texto)
    texto = texto.lower()
    texto = remover_pontuacao(texto)
    return normalizar_espacos(texto)


# ── NLTK e spaCy (seções 4.3.2.1 e 4.3.2.2) ────────────────────────────────

_nlp = None
_stopwords = None


def _carregar_spacy():
    """Carrega o modelo de português sob demanda e o mantém em cache.

    Desativa parser e NER: só a lematização é usada aqui, e desligar o
    resto acelera bastante o processamento em lote.
    """
    global _nlp
    if _nlp is None:
        import spacy
        try:
            _nlp = spacy.load('pt_core_news_sm', disable=['parser', 'ner'])
        except OSError as erro:
            raise OSError(
                "Modelo do spaCy ausente. No Colab ele precisa ser baixado "
                "a cada runtime:\n"
                "    !python -m spacy download pt_core_news_sm -q"
            ) from erro
    return _nlp


def _carregar_stopwords():
    """Stopwords do NLTK para português, baixadas sob demanda."""
    global _stopwords
    if _stopwords is None:
        import nltk
        try:
            from nltk.corpus import stopwords
            _stopwords = set(stopwords.words('portuguese'))
        except LookupError:
            nltk.download('stopwords', quiet=True)
            from nltk.corpus import stopwords
            _stopwords = set(stopwords.words('portuguese'))
    return _stopwords


def tokenizar(texto):
    """Tokenização com NLTK (seção 4.3.2.1)."""
    import nltk
    try:
        return nltk.word_tokenize(texto, language='portuguese')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)
        nltk.download('punkt', quiet=True)
        return nltk.word_tokenize(texto, language='portuguese')


def lematizar(textos):
    """Lematização em lote com spaCy (seção 4.3.2.2).

    Recebe e devolve listas — `nlp.pipe` é ordens de magnitude mais rápido
    que chamar o modelo mensagem a mensagem.
    """
    nlp = _carregar_spacy()
    return [
        ' '.join(token.lemma_ for token in doc if not token.is_space)
        for doc in nlp.pipe(list(textos), batch_size=64)
    ]


def remover_stopwords(texto):
    """Remove stopwords do português.

    ATENÇÃO — não é o padrão do pipeline, e por um motivo: SMS têm 15 a 25
    palavras, e remover stopwords de uma mensagem tão curta descarta boa
    parte do sinal. Além disso, marcadores de urgência e de comando
    ('já', 'agora', 'você', 'sua') estão na lista de stopwords e são
    exatamente o vocabulário do golpe.

    Use como variante experimental, compare na validação, e reporte as duas
    versões — a comparação em si já é resultado.
    """
    stop = _carregar_stopwords()
    return ' '.join(p for p in texto.split() if p not in stop)


def preparar_classico(textos, lematizacao=True, stopwords=False):
    """Pipeline completo de entrada dos modelos clássicos.

    Parâmetros
    ----------
    lematizacao : bool
        Aplica spaCy. Padrão ligado.
    stopwords : bool
        Remove stopwords. Padrão DESLIGADO — ver `remover_stopwords`.
    """
    saida = [normalizar(t) for t in textos]

    if lematizacao:
        saida = lematizar(saida)

    if stopwords:
        saida = [remover_stopwords(t) for t in saida]

    return saida


# ── Features artesanais ────────────────────────────────────────────────────
# Complementam o TF-IDF com sinais que a bag-of-words não captura bem.
# Calculadas sobre o texto ORIGINAL — proporção de maiúsculas e presença
# de '!' se perdem depois da normalização.

_URGENCIA = re.compile(
    r'\b(urgente|imediat\w*|agora|hoje|ultim\w+|expir\w+|venc\w+|'
    r'bloque\w+|suspens\w+|cancel\w+|prazo|atencao|atenção|aviso|'
    r'irreversivel|irreversível|24\s*horas|ultima\s+chance)\b',
    re.IGNORECASE,
)

_INSTITUICAO = re.compile(
    r'\b(banco|caixa|bradesco|itau|itaú|santander|nubank|bb|'
    r'inss|receita|serasa|spc|detran|correios|gov|governo|'
    r'previdencia|previdência|pix|cpf|conta|cartao|cartão)\b',
    re.IGNORECASE,
)

_PREMIO = re.compile(
    r'\b(premi\w+|sorte\w+|ganho\w*|ganhou|contempla\w+|'
    r'restitui\w+|reembols\w+|credito\s+liberado|crédito\s+liberado|'
    r'saque|liberad\w+|beneficio|benefício|gratis|grátis)\b',
    re.IGNORECASE,
)

_ACAO = re.compile(
    r'\b(clique|acesse|confirme|atualize|regularize|valide|'
    r'informe|envie|responda|baixe|instale|cadastre)\b',
    re.IGNORECASE,
)


def extrair_features(texto):
    """Features artesanais de uma mensagem. Devolve um dicionário."""
    if not isinstance(texto, str):
        texto = ''

    n_chars = len(texto)
    letras = [c for c in texto if c.isalpha()]

    return {
        'tem_url':        int(bool(_URL.search(texto))),
        'tem_email':      int(bool(_EMAIL.search(texto))),
        'tem_telefone':   int(bool(_TELEFONE.search(texto))),
        'tem_valor':      int(bool(_VALOR.search(texto))),
        'tem_urgencia':   int(bool(_URGENCIA.search(texto))),
        'tem_instituicao': int(bool(_INSTITUICAO.search(texto))),
        'tem_premio':     int(bool(_PREMIO.search(texto))),
        'tem_acao':       int(bool(_ACAO.search(texto))),
        'n_chars':        n_chars,
        'n_palavras':     len(texto.split()),
        'n_exclamacoes':  texto.count('!'),
        # Proporção de maiúsculas: caixa alta é recurso de urgência visual
        'prop_maiusculas': (
            sum(1 for c in letras if c.isupper()) / len(letras) if letras else 0.0
        ),
    }


def matriz_features(textos):
    """Aplica `extrair_features` a uma série e devolve um DataFrame."""
    import pandas as pd
    return pd.DataFrame([extrair_features(t) for t in textos])


# ── Deduplicação ───────────────────────────────────────────────────────────

def chave_dedup(texto):
    """Chave de deduplicação: texto normalizado, sem espaços.

    Deduplicar por texto exato deixa passar variações triviais de
    espaçamento, pontuação e caixa. Como o corpus reúne fontes que se
    sobrepõem, a comparação precisa ser sobre a forma normalizada.
    """
    return normalizar(texto).replace(' ', '')


def id_mensagem(texto):
    """Identidade de uma mensagem, derivada do próprio conteúdo.

    Usada para ligar uma mensagem sintética à mensagem real que a originou
    (`id_semente`), através de notebooks e arquivos diferentes.

    Precisa ser derivada do CONTEÚDO, não da posição na planilha. Um id
    posicional (`semente_3`, `bortot_3`) depende da ordem das linhas e do
    arquivo em que a mensagem foi lida: o notebook 00 lê as sementes de um
    CSV e o notebook 01 lê o corpus de outro, então dois ids posicionais
    para a mesma mensagem nunca coincidem — e o filtro anti-vazamento passa
    a descartar toda a augmentation sem que nada acuse o erro.

    Sobre `chave_dedup`, e não sobre o texto cru: assim a identidade
    sobrevive a diferenças de espaçamento, pontuação e caixa entre as cópias
    da mesma mensagem em fontes distintas — a mesma tolerância que a
    deduplicação já assume.
    """
    chave = chave_dedup(texto).encode('utf-8')
    return 'msg_' + hashlib.sha1(chave).hexdigest()[:12]
