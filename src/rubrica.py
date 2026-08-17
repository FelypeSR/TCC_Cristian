"""
Rubrica de risco para classificação por LLM — notebook 04.

A seção 2 (Justificativa) e a seção 5 da monografia prometem uma rubrica
que atribui **pontuação e níveis de risco**, não apenas um rótulo binário.
Este módulo implementa isso no espírito de Dimario, Bacha e Butka (2024).

Como funciona: em vez de pedir ao modelo uma probabilidade — que um modelo
de 3B calibra mal — pede-se que ele marque **quais critérios de golpe estão
presentes** na mensagem. A pontuação é então calculada de forma determinística
a partir dos critérios marcados. Isso traz três vantagens:

  1. Interpretabilidade — dá para dizer ao usuário POR QUE a mensagem é
     suspeita, que é o que sustenta o uso educativo defendido no TCC e
     alimenta as diretrizes da etapa 4.4.5.
  2. Robustez — marcar presença de um padrão é tarefa mais fácil para um
     modelo pequeno do que emitir um número calibrado.
  3. Determinismo — a pontuação não depende da calibração interna do modelo.

O limiar que separa smishing de legítima NÃO é fixado aqui: é calibrado na
validação por F2 (ver `evaluation.calibrar_limiar`).
"""

import re

import config as CFG

# ── Critérios ──────────────────────────────────────────────────────────────
# Ancorados no referencial teórico da própria monografia: Bortot et al.
# (2024, p. 13), citado na seção 3.2, e os padrões descritos na seção 3.1.
# Manter essa ancoragem é o que diferencia uma rubrica fundamentada de uma
# lista de critérios inventada — e é defensável em banca.

CRITERIOS = [
    {
        'id': 1,
        'nome': 'identidade_falsa',
        'descricao': 'A mensagem se apresenta como sendo de banco, plano de saúde, '
                     'órgão do governo ou empresa conhecida.',
        'explicacao': 'A mensagem finge vir de uma instituição confiável.',
        'origem': 'Bortot et al. (2024, p. 13)',
    },
    {
        'id': 2,
        'nome': 'urgencia',
        'descricao': 'Cria urgência ou pressão de tempo: prazo curto, "verificação '
                     'imediata", ameaça de perder algo se não agir agora.',
        'explicacao': 'Cria pressa para você não ter tempo de desconfiar.',
        'origem': 'Bortot et al. (2024, p. 13)',
    },
    {
        'id': 3,
        'nome': 'link_ou_acao',
        'descricao': 'Pede para clicar em um link, baixar um aplicativo ou acessar '
                     'um endereço na internet.',
        'explicacao': 'Pede que você clique em um link ou instale algo.',
        'origem': 'Bortot et al. (2024, p. 13)',
    },
    {
        'id': 4,
        'nome': 'premio',
        'descricao': 'Oferece prêmio, sorteio, restituição, crédito liberado ou '
                     'benefício inesperado.',
        'explicacao': 'Promete um ganho que você não estava esperando.',
        'origem': 'Bortot et al. (2024, p. 13)',
    },
    {
        'id': 5,
        'nome': 'pedido_dados',
        'descricao': 'Solicita dados pessoais ou bancários: senha, CPF, número do '
                     'cartão, código de verificação, PIX.',
        'explicacao': 'Pede dados pessoais ou bancários seus.',
        'origem': 'Seção 3.1 da monografia',
    },
    {
        'id': 6,
        'nome': 'ameaca_perda',
        'descricao': 'Ameaça com perda ou punição: bloqueio de conta, cancelamento '
                     'de benefício, dívida, nome sujo, processo.',
        'explicacao': 'Ameaça você com bloqueio, dívida ou perda de benefício.',
        'origem': 'Seção 3.2 da monografia',
    },
    {
        'id': 7,
        'nome': 'emergencia_familiar',
        'descricao': 'Simula emergência com familiar ou conhecido, geralmente pedindo '
                     'dinheiro de um número desconhecido.',
        'explicacao': 'Finge ser um familiar em apuros pedindo dinheiro.',
        'origem': 'Seção 3.2 da monografia',
    },
]

CRITERIOS_POR_ID = {c['id']: c for c in CRITERIOS}
N_CRITERIOS = len(CRITERIOS)


# ── Prompt ─────────────────────────────────────────────────────────────────

def _bloco_criterios():
    return '\n'.join(f"{c['id']}. {c['descricao']}" for c in CRITERIOS)


SYSTEM_PROMPT = f"""Você analisa mensagens SMS recebidas por pessoas idosas no Brasil e identifica indícios de golpe (smishing).

Avalie a mensagem segundo os critérios abaixo:

{_bloco_criterios()}

Responda APENAS com os números dos critérios presentes na mensagem, separados por vírgula.
Se nenhum critério estiver presente, responda exatamente: NENHUM

Não escreva explicações, não repita a mensagem, não acrescente nenhum outro texto."""


def construir_mensagens(texto_alvo, exemplos=None):
    """Monta a conversa no formato de chat esperado pelo Llama 3.

    `exemplos` é um DataFrame com as colunas de texto e de critérios, vindo
    EXCLUSIVAMENTE do conjunto de treino — nunca de validação ou teste.
    """
    mensagens = [{'role': 'system', 'content': SYSTEM_PROMPT}]

    if exemplos is not None:
        for _, linha in exemplos.iterrows():
            mensagens.append({'role': 'user', 'content': f'Mensagem: {linha["texto"]}'})
            mensagens.append({'role': 'assistant', 'content': linha['resposta_esperada']})

    mensagens.append({'role': 'user', 'content': f'Mensagem: {texto_alvo}'})
    return mensagens


def resposta_esperada(criterios):
    """Formata a resposta de referência de um exemplo few-shot."""
    if not criterios:
        return 'NENHUM'
    return ','.join(str(c) for c in sorted(criterios))


# ── Parsing ────────────────────────────────────────────────────────────────

_NUMEROS = re.compile(r'\d+')

# "3 critérios", "2 criterios presentes" — a CONTAGEM de critérios, não um id.
# Sem remover isso antes de varrer os dígitos, a resposta "a mensagem tem 3
# critérios: 1, 2" seria lida como {1, 2, 3}: a contagem entraria na lista e
# inflaria a pontuação de risco.
_CONTAGEM = re.compile(r'\d+\s*crit[eé]rios?', re.IGNORECASE)


def parse_resposta(bruta):
    """Extrai os critérios marcados da resposta do modelo.

    Devolve (criterios, valida):
      - criterios: lista de ids reconhecidos
      - valida   : False quando o modelo não seguiu o formato

    Um modelo de 3B ignora a instrução de formato com alguma frequência, e
    cada modo de desvio erra para um lado diferente:

      "a mensagem tem 3 critérios: 1, 2"   → a contagem viraria critério
      "nenhum dos 7 critérios"             → o 7 viraria critério
      "nenhum critério, exceto o 2 e o 3"  → a negação engoliria a lista

    Os dois primeiros são recuperáveis descartando a contagem ("N critérios")
    antes de varrer os dígitos e, havendo dois-pontos, lendo só o que vem
    depois deles. O terceiro não é: uma negação que sobrevive à limpeza e
    ainda convive com números é genuinamente ambígua, e adivinhar ali trocaria
    um erro visível por um invisível. Esse caso vira abstenção.

    Quem decide o que fazer com uma resposta inválida é o notebook. A
    recomendação é contá-la como abstenção e reportar a taxa — e, se for
    preciso atribuir um rótulo, atribuir SMISHING. Cair no rótulo negativo
    produziria justamente o erro mais caro do projeto.
    """
    if not isinstance(bruta, str):
        return [], False

    texto = bruta.strip().lower()
    if not texto:
        return [], False

    nega = 'nenhum' in texto

    # A contagem sai antes de qualquer varredura de dígitos
    segmento = _CONTAGEM.sub(' ', texto)

    # Havendo dois-pontos, a lista está depois deles: em "critérios
    # presentes: 1, 2" tudo que interessa vem à direita.
    if ':' in segmento:
        segmento = segmento.rsplit(':', 1)[1]

    encontrados = [int(n) for n in _NUMEROS.findall(segmento)]
    validos = sorted({n for n in encontrados if n in CRITERIOS_POR_ID})

    # Negação junto com números: indistinguível entre "nenhum, exceto o 2" e
    # "nenhum dos 7". Abstenção é a leitura honesta.
    if nega and validos:
        return [], False

    if nega:
        return [], True

    # Sem negação e sem número válido, o modelo respondeu outra coisa
    return validos, bool(validos)


def pontuar(criterios):
    """Pontuação de risco: proporção de critérios presentes, em [0, 1].

    Pesos uniformes por decisão deliberada. Atribuir pesos diferentes a cada
    critério exigiria evidência empírica que este trabalho não tem — e um
    peso arbitrário é exatamente o tipo de escolha que a banca questiona.
    A assimetria de custo entre os erros é tratada no limiar, calibrado na
    validação por F2, e não em pesos escolhidos a dedo.
    """
    return len(set(criterios)) / N_CRITERIOS


def classificar_resposta(bruta):
    """Pipeline completo de uma resposta bruta do modelo.

    Devolve um dicionário com critérios, pontuação, nível de risco e a
    marcação de validade. O rótulo binário NÃO sai daqui: depende do limiar
    calibrado na validação, aplicado depois pelo notebook.
    """
    criterios, valida = parse_resposta(bruta)
    score = pontuar(criterios)
    return {
        'criterios': criterios,
        'criterios_nomes': [CRITERIOS_POR_ID[c]['nome'] for c in criterios],
        'n_criterios': len(criterios),
        'score': score,
        'nivel_risco': CFG.nivel_risco(score),
        'resposta_valida': valida,
        'resposta_bruta': bruta,
    }


# ── Saída para o usuário final (etapa 4.4.5) ───────────────────────────────

def explicar(criterios, nivel=None):
    """Texto em linguagem simples explicando por que a mensagem é suspeita.

    É o insumo direto das diretrizes da etapa 4.4.5: um alerta que apenas
    diz "golpe" não educa ninguém. Dizer QUAL indício foi encontrado é o que
    dá ao trabalho o caráter preventivo e educativo que a monografia defende
    do começo ao fim.
    """
    if not criterios:
        return 'Nenhum indício de golpe foi encontrado nesta mensagem.'

    nivel = nivel or CFG.nivel_risco(pontuar(criterios))
    cabecalho = {
        'baixo': 'Esta mensagem tem poucos indícios de golpe, mas fique atento:',
        'medio': 'Atenção: esta mensagem tem características comuns em golpes:',
        'alto':  'CUIDADO: esta mensagem tem várias características de golpe:',
    }[nivel]

    itens = '\n'.join(f'  • {CRITERIOS_POR_ID[c]["explicacao"]}' for c in sorted(criterios))
    rodape = ('\nNa dúvida, não clique em nada e procure a instituição pelo '
              'telefone oficial que você já conhece.')
    return f'{cabecalho}\n{itens}\n{rodape}'


def tabela_criterios():
    """Tabela dos critérios com a origem de cada um, para a monografia."""
    import pandas as pd
    return pd.DataFrame([
        {'#': c['id'], 'Critério': c['nome'], 'Descrição': c['descricao'], 'Origem': c['origem']}
        for c in CRITERIOS
    ])
