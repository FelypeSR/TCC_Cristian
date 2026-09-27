"""
Monta os CSVs de data/raw/ a partir da pasta tccdados/.

Saídas (em tccdados/raw/, para upload em data/raw/ no Drive):
    mensagens_coletadas.csv   fonte 'coletada' do notebook 01
    mensagens_simuladas.csv   fonte 'simulada' do notebook 01
    sementes_pt.csv           sementes de smishing do notebook 00
    kaggle_traduzido.csv      só validação de pipeline — NÃO entra em FONTES_PT

Decisões de curadoria:
  - Mensagens de ligação (canal='ligacao') ficam de fora: não são smishing.
  - Os prints do WhatsApp em tccdados/ reproduzem mensagens já presentes nos
    CSVs (8 do sintético, 20 do Kaggle) e não geram linhas novas.
  - O placeholder '<URL>' do corpus sintético é trocado por um domínio. Sem
    isso, o regex de URL do preprocessing não o reconhece e a palavra 'url'
    vira um marcador exclusivo das sintéticas de golpe.
  - Linhas com rótulo 'suspeita' ficam de fora até serem rotuladas.

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

TIPO_POR_ID_COLETADO = {'62': 'outro', '65': 'outro'}


def rotulo(valor):
    return CFG.MAPA_ROTULOS.get(valor.strip().lower())


def ler(caminho):
    with open(caminho, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def escrever(caminho, linhas, colunas):
    with open(caminho, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=colunas)
        w.writeheader()
        w.writerows(linhas)
    print(f'{os.path.basename(caminho)}: {len(linhas)} linhas')


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

    coletadas, kaggle = [], []
    for r in ler(os.path.join(pasta, 'mensagens_rows.csv')):
        rot = rotulo(r['classificacao'])
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

    colunas = ['mensagem', 'rotulo', 'tipo_golpe', 'id_original', 'canal']
    escrever(os.path.join(saida, 'mensagens_simuladas.csv'), simuladas, colunas)
    escrever(os.path.join(saida, 'mensagens_coletadas.csv'), coletadas, colunas)
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
