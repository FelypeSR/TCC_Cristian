"""
Monta os CSVs de data/raw/ a partir da pasta tccdados/.

Saídas (em tccdados/raw/, para upload em data/raw/ no Drive):
    mensagens_coletadas.csv   fonte 'coletada' do notebook 01
    mensagens_simuladas.csv   fonte 'simulada' do notebook 01
    mensagens_traduzidas.csv  fonte 'traduzida' do notebook 01
    sementes_pt.csv           sementes de smishing do notebook 00
    kaggle_traduzido.csv      as 60 mensagens traduzidas, sem filtro — NÃO entra
                              em FONTES_PT (o subconjunto aprovado é o arquivo
                              mensagens_traduzidas.csv)

Entrada opcional:
    rascunho_legitimas_simuladas.csv   legítimas simuladas em revisão. Só as
    linhas com aprovada=1 entram em mensagens_simuladas.csv.

Decisões de curadoria:
  - Mensagens de ligação (canal='ligacao') ficam de fora: não são smishing.
  - Os prints do WhatsApp em tccdados/ reproduzem mensagens já presentes nos
    CSVs (8 do sintético, 20 do Kaggle) e não geram linhas novas.
  - O placeholder '<URL>' do corpus sintético é trocado por um domínio. Sem
    isso, o regex de URL do preprocessing não o reconhece e a palavra 'url'
    vira um marcador exclusivo das sintéticas de golpe.
  - Linhas com rótulo 'suspeita' ficam de fora até serem rotuladas.
  - A id 62 (código de verificação do WhatsApp) entra como legítima, e não
    como fraude, que é o rótulo de mensagens_rows.csv; a id 68 (Santander,
    renegociação pelo IR) entra como smishing, e não como legítima (ver
    ROTULO_POR_ID_COLETADO).
  - Das 60 mensagens traduzidas, entram 22: as de SMS e WhatsApp sem marca
    estrangeira (rupia, dólar, Bengaluru, EPFO, IPL). E-mail está fora do escopo.
    Entram 15 legítimas E 7 de smishing: se só as legítimas entrassem, o estilo
    de texto traduzido viraria um atalho para o rótulo. A id 3 é tradução direta
    de uma mensagem dos datasets em inglês e fica de fora.

Uso:
    python tools/montar_fontes_pt.py <pasta tccdados>
"""

import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import config as CFG

# categoria do corpus sintético -> CFG.TIPOS_GOLPE
MAPA_CATEGORIA = {
    'inss_beneficio':  'beneficio',
    'banco_pix':       'banco',
    'falso_parente':   'emergencia',
    'falso_sequestro': 'emergencia',
    'orgao_publico':   'orgao_publico',
    'servico_entrega': 'entrega',
    'cobranca_boleto': 'outro',
    'consignado':      'credito',
    'suporte_tecnico': 'outro',
    'premio_sorteio':  'premio',
}

# Mensagens cuja categoria original não descreve o golpe
TIPO_POR_ID = {
    'sint_0023': 'outro',       # corte de energia, estava em servico_entrega
    'sint_0024': 'beneficio',   # Farmácia Popular, estava em servico_entrega
    'sint_0033': 'outro',       # código de verificação, estava em premio_sorteio
}

# Domínios exibidos nos prints de simulação do WhatsApp
DOMINIO_PRINT = {
    'sint_0002': 'gov-inss-atualiza.com',
    'sint_0003': 'meuinss-retroativo.net',
    'sint_0007': 'banco-renovacao.site/senha',
    'sint_0019': 'restituicao-irpf.site',
    'sint_0032': 'sorteio-premiado.site',
}

# mensagens_rows.csv: ids de coleta própria (as demais vêm do Kaggle traduzido)
IDS_COLETADOS = {'62', '63', '65', '68', '70'}
IDS_KAGGLE = {str(i) for i in range(1, 61)}

TIPO_POR_ID_COLETADO = {'65': 'outro', '68': 'banco'}

# Rótulos revistos depois da coleta (decisões de 2026-11-08).
#   62: SMS real com o código de verificação do WhatsApp, sem link nem pedido.
#       Estava como fraude porque chegou durante uma tentativa de golpe, mas
#       rotulada assim ensina que o código legítimo do WhatsApp é golpe.
#   68: "SANTANDER" oferecendo renegociação de dívidas pela "nova regra do IR
#       2026", com link. Estava como legítima, mas com tipo_golpe=smishing na
#       própria linha. Bancos não costumam mandar mensagem sobre IR com link;
#       o domínio oficial no texto não basta, porque o remetente não foi
#       registrado.
ROTULO_POR_ID_COLETADO = {'62': 'legitima', '68': 'smishing'}

# mensagens_rows.csv: subconjunto traduzido que entra no corpus (ver curadoria)
IDS_TRADUZIDAS_LEGITIMAS = {
    '31', '32', '34', '35', '37', '42', '43', '45',
    '47', '48', '50', '51', '52', '54', '55',
}
TIPO_TRADUZIDA_SMISHING = {
    '2': 'premio', '8': 'credito', '21': 'entrega', '25': 'credito',
    '26': 'outro', '28': 'outro', '29': 'banco',
}
IDS_TRADUZIDAS = IDS_TRADUZIDAS_LEGITIMAS | set(TIPO_TRADUZIDA_SMISHING)

# Vírgulas que a tradução automática trocou por ponto
CORRECOES_TRADUCAO = {
    '34': ('maravilhoso. repleto', 'maravilhoso, repleto'),
    '51': ('sexta-feira. às 13h', 'sexta-feira, às 13h'),
}


def rotulo(valor):
    return CFG.MAPA_ROTULOS.get(valor.strip().lower())


def ler(caminho):
    # utf-8-sig: planilhas podem salvar o CSV com BOM, o que estragaria o nome
    # da primeira coluna
    with open(caminho, encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def escrever(caminho, linhas, colunas):
    with open(caminho, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=colunas)
        w.writeheader()
        w.writerows(linhas)
    print(f'{os.path.basename(caminho)}: {len(linhas)} linhas')


def traduzida(r):
    """Linha de mensagens_traduzidas.csv a partir de mensagens_rows.csv."""
    # split/join também tira o espaço duplo deixado pelos valores perdidos na tradução
    texto = ' '.join(r['texto'].split())
    if r['id'] in CORRECOES_TRADUCAO:
        antigo, novo = CORRECOES_TRADUCAO[r['id']]
        assert antigo in texto, f"correção da id {r['id']} não se aplica"
        texto = texto.replace(antigo, novo)
    return {
        'mensagem': texto,
        'rotulo': rotulo(r['classificacao']),
        'tipo_golpe': TIPO_TRADUZIDA_SMISHING.get(r['id'], 'nao_aplicavel'),
        'id_original': f"kaggle_{r['id']}",
        'canal': r['fonte'].lower(),
    }


def legitimas_aprovadas(pasta, colunas):
    """Legítimas simuladas do rascunho com aprovada=1. Vazia se não houver."""
    caminho = os.path.join(pasta, 'rascunho_legitimas_simuladas.csv')
    if not os.path.isfile(caminho):
        return []
    rascunho = ler(caminho)
    aprovadas = [r for r in rascunho if r['aprovada'].strip() == '1']
    print(f'rascunho de legítimas: {len(aprovadas)} aprovadas de {len(rascunho)}')
    return [{c: r[c] for c in colunas} for r in aprovadas]


def main(pasta):
    saida = os.path.join(pasta, 'raw')
    os.makedirs(saida, exist_ok=True)

    simuladas = []
    for r in ler(os.path.join(pasta, 'corpus_sintetico_smishing.csv')):
        if r['canal'] == 'ligacao':
            continue
        texto = r['texto']
        if '<URL>' in texto:
            dominio = DOMINIO_PRINT.get(r['id'])
            if not dominio:
                dominio = r['dominio_ficticio'].replace('.invalid', '.site')
            texto = texto.replace('<URL>', dominio)
        simuladas.append({
            'mensagem': texto,
            'rotulo': rotulo(r['label_nome']),
            'tipo_golpe': TIPO_POR_ID.get(r['id'], MAPA_CATEGORIA[r['categoria']]),
            'id_original': r['id'],
            'canal': r['canal'],
        })

    coletadas, kaggle, traduzidas = [], [], []
    for r in ler(os.path.join(pasta, 'mensagens_rows.csv')):
        rot = rotulo(ROTULO_POR_ID_COLETADO.get(r['id'], r['classificacao']))
        if rot is None:
            print(f"  [fora] id {r['id']}: rótulo '{r['classificacao']}'")
            continue
        tipo = 'nao_aplicavel' if rot == CFG.CLASSE_NEGATIVA else 'outro'
        if r['id'] in IDS_COLETADOS:
            coletadas.append({
                'mensagem': r['texto'],
                'rotulo': rot,
                'tipo_golpe': TIPO_POR_ID_COLETADO.get(r['id'], tipo),
                'id_original': r['id'],
                'canal': r['fonte'].lower(),
            })
        elif r['id'] in IDS_KAGGLE:
            kaggle.append({'texto': r['texto'], 'rotulo': rot, 'tipo_golpe': tipo})
            if r['id'] in IDS_TRADUZIDAS:
                traduzidas.append(traduzida(r))

    colunas = ['mensagem', 'rotulo', 'tipo_golpe', 'id_original', 'canal']
    simuladas += legitimas_aprovadas(pasta, colunas)
    escrever(os.path.join(saida, 'mensagens_simuladas.csv'), simuladas, colunas)
    escrever(os.path.join(saida, 'mensagens_coletadas.csv'), coletadas, colunas)
    escrever(os.path.join(saida, 'mensagens_traduzidas.csv'), traduzidas, colunas)
    escrever(os.path.join(saida, 'kaggle_traduzido.csv'), kaggle,
             ['texto', 'rotulo', 'tipo_golpe'])

    # Notebook 00 lê a coluna 'texto'; o texto é idêntico ao de FONTES_PT,
    # então o id_semente (hash do conteúdo) casa no notebook 01.
    sementes = [
        {'texto': m['mensagem'], 'rotulo': m['rotulo'], 'tipo_golpe': m['tipo_golpe']}
        for m in coletadas + simuladas if m['rotulo'] == CFG.CLASSE_POSITIVA
    ]
    escrever(os.path.join(saida, 'sementes_pt.csv'), sementes,
             ['texto', 'rotulo', 'tipo_golpe'])


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'tccdados')
