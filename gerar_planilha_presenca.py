import argparse
import os
import sqlite3
import tkinter as tk
from tkinter import ttk
from typing import Dict, List

import pandas as pd


DB_NAME = "rostos.db"
TABELA_PADRAO = "rostos"
PASTA_PLANILHAS = "planilhas"
MAX_TEXTO = 80
MAX_BYTES_HEX = 24


def truncar_texto(valor, limite=MAX_TEXTO):
    texto = str(valor)
    if len(texto) <= limite:
        return texto
    return f"{texto[:limite]}..."


def resumir_valor_grande(valor):
    """Resumo legível para campos grandes (ex.: BLOBs de rosto/embeddings)."""
    if valor is None:
        return ""

    if isinstance(valor, (bytes, bytearray, memoryview)):
        dados = bytes(valor)
        prefixo = dados[:MAX_BYTES_HEX].hex()
        if len(dados) > MAX_BYTES_HEX:
            return f"<BLOB {len(dados)} bytes> {prefixo}..."
        return f"<BLOB {len(dados)} bytes> {prefixo}"

    if isinstance(valor, str):
        return truncar_texto(valor)

    return valor


def carregar_tabela_para_planilha(db_name=DB_NAME, tabela=TABELA_PADRAO):
    if not os.path.exists(db_name):
        raise FileNotFoundError(
            f"Banco de dados '{db_name}' não encontrado."
        )

    conn = sqlite3.connect(db_name)
    try:
        df = pd.read_sql_query(f"SELECT * FROM {tabela}", conn)
    finally:
        conn.close()

    if df.empty:
        return df

    # Compatibilidade com base legada: consolida em uma coluna única.
    if 'embedding' in df.columns:
        df = df.drop(columns=['embedding'])

    if 'rosto' in df.columns and 'rosto_embeddings' not in df.columns:
        df = df.rename(columns={'rosto': 'rosto_embeddings'})

    # Aplica resumo em todos os campos para evitar células gigantes.
    for coluna in df.columns:
        df[coluna] = df[coluna].apply(resumir_valor_grande)

    return df


def _normalizar_texto(valor):
    if valor is None:
        return ""
    texto = str(valor).strip()
    if not texto:
        return ""
    return texto


def _ordenar_registros(df):
    df_ordenado = df.copy()
    if 'data_imagem' in df_ordenado.columns:
        df_ordenado['data_imagem_dt'] = pd.to_datetime(df_ordenado['data_imagem'], errors='coerce')
    else:
        df_ordenado['data_imagem_dt'] = pd.NaT

    ordem_colunas = []
    for coluna in ['data_imagem_dt', 'numero_imagem', 'ord_id']:
        if coluna in df_ordenado.columns:
            ordem_colunas.append(coluna)

    if ordem_colunas:
        df_ordenado = df_ordenado.sort_values(by=ordem_colunas, ascending=True, na_position='last')

    return df_ordenado


def construir_planilha_modelo_presenca(df_detalhado):
    """
    Monta a planilha no formato de presenca por pessoa.
    Cada nova imagem cadastrada adiciona novas colunas com nomes literais
    do banco: numero_imagem e data_imagem.
    """
    if df_detalhado is None or df_detalhado.empty:
        return pd.DataFrame()

    df_base = df_detalhado.copy()

    obrigatorias = ['id_rosto', 'nome', 'turma', 'numero_imagem', 'data_imagem']
    faltantes = [col for col in obrigatorias if col not in df_base.columns]
    if faltantes:
        raise ValueError(f"Colunas obrigatórias ausentes para consolidar presença: {faltantes}")

    df_base['nome_limpo'] = df_base['nome'].apply(_normalizar_texto)
    df_base['id_rosto'] = df_base['id_rosto'].apply(_normalizar_texto)
    df_base['turma'] = df_base['turma'].apply(_normalizar_texto)
    df_base['data_imagem'] = df_base['data_imagem'].apply(_normalizar_texto)

    # Regra de identidade:
    # - com nome: agrupa por nome (reconhecidos/manualmente nomeados)
    # - sem nome: cada id_rosto vira uma linha separada
    df_base['identificador'] = df_base.apply(
        lambda linha: linha['nome_limpo'] if linha['nome_limpo'] else f"SEM_NOME::{linha['id_rosto']}",
        axis=1,
    )

    registros: List[Dict[str, object]] = []

    # Colunas dinamicas globais por imagem (modelo solicitado).
    imagens_distintas = []
    for valor in df_base['numero_imagem'].tolist():
        if pd.isna(valor) or str(valor).strip() == '':
            continue
        try:
            numero = int(valor)
        except (ValueError, TypeError):
            continue
        imagens_distintas.append(numero)
    imagens_distintas = sorted(set(imagens_distintas))

    for identificador, grupo in df_base.groupby('identificador', sort=False):
        grupo = _ordenar_registros(grupo)

        nomes_validos = [n for n in grupo['nome_limpo'].tolist() if n]
        nome_exibicao = nomes_validos[0] if nomes_validos else ''

        turmas_validas = [t for t in grupo['turma'].tolist() if t]
        turma = turmas_validas[0] if turmas_validas else ''

        ids_vinculados = [i for i in grupo['id_rosto'].tolist() if i]
        linha = {
            'id_rosto': ', '.join(ids_vinculados),
            'nome': nome_exibicao,
            'turma': turma,
        }

        # Preenche colunas por imagem para evitar repeticao de linhas por nome.
        # Se houver mais de um registro da mesma pessoa na mesma imagem,
        # mantemos a primeira data registrada dessa imagem.
        grupo_por_imagem = (
            grupo.sort_values(by=['data_imagem_dt', 'ord_id'], ascending=True, na_position='last')
            .drop_duplicates(subset=['numero_imagem'], keep='first')
        )

        mapa_imagem_data = {
            int(row['numero_imagem']): row['data_imagem']
            for _, row in grupo_por_imagem.iterrows()
            if pd.notna(row['numero_imagem']) and str(row['numero_imagem']).strip() != ''
        }

        ocorrencias = []
        for numero_img in imagens_distintas:
            if numero_img in mapa_imagem_data:
                ocorrencias.append(numero_img)
                ocorrencias.append(mapa_imagem_data[numero_img])
            else:
                ocorrencias.append('')
                ocorrencias.append('')

        linha['_ocorrencias'] = ocorrencias

        registros.append(linha)

    if not registros:
        return pd.DataFrame()

    colunas = ['id_rosto', 'nome', 'turma']
    for _ in imagens_distintas:
        colunas.append('numero_imagem')
        colunas.append('data_imagem')

    linhas = []
    for item in registros:
        linha = [item['id_rosto'], item['nome'], item['turma']]
        linha.extend(item['_ocorrencias'])
        linhas.append(linha)

    df_modelo = pd.DataFrame(linhas, columns=colunas)
    df_modelo = df_modelo.sort_values(by=['nome', 'id_rosto'], ascending=True).reset_index(drop=True)
    return df_modelo


def exibir_planilha_em_janela(df, titulo="Planilha do Banco de Dados"):
    if df is None or df.empty:
        print("Nenhum dado para exibir na planilha.")
        return

    root = tk.Tk()
    root.title(titulo)
    root.geometry("1400x720")

    frame = ttk.Frame(root, padding=8)
    frame.pack(fill="both", expand=True)

    colunas = list(df.columns)
    tree = ttk.Treeview(frame, columns=colunas, show="headings")

    scroll_y = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    scroll_x = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
    tree.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)

    tree.grid(row=0, column=0, sticky="nsew")
    scroll_y.grid(row=0, column=1, sticky="ns")
    scroll_x.grid(row=1, column=0, sticky="ew")

    frame.columnconfigure(0, weight=1)
    frame.rowconfigure(0, weight=1)

    for coluna in colunas:
        tree.heading(coluna, text=coluna)
        largura = max(120, min(320, len(coluna) * 12))
        tree.column(coluna, width=largura, minwidth=80, anchor="center")

    for _, linha in df.iterrows():
        valores = ["" if pd.isna(valor) else str(valor) for valor in linha.tolist()]
        tree.insert("", "end", values=valores)

    root.mainloop()


def gerar_planilha(db_name=DB_NAME, tabela=TABELA_PADRAO, saida_base="planilha_presenca"):
    df = carregar_tabela_para_planilha(db_name=db_name, tabela=tabela)

    if df.empty:
        print(f"A tabela '{tabela}' está vazia.")
        return df

    df_modelo = construir_planilha_modelo_presenca(df)

    # Garante pasta única para organizar os arquivos gerados.
    os.makedirs(PASTA_PLANILHAS, exist_ok=True)

    # Limpa arquivos legados para manter apenas a planilha de presença.
    legado_csv = os.path.join(PASTA_PLANILHAS, f"{saida_base}_banco.csv")
    legado_xlsx = os.path.join(PASTA_PLANILHAS, f"{saida_base}_banco.xlsx")
    if os.path.exists(legado_csv):
        os.remove(legado_csv)
    if os.path.exists(legado_xlsx):
        os.remove(legado_xlsx)

    csv_path = os.path.join(PASTA_PLANILHAS, f"{saida_base}.csv")
    df_modelo.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"Planilha CSV gerada em: {csv_path}")

    xlsx_path = os.path.join(PASTA_PLANILHAS, f"{saida_base}.xlsx")
    try:
        df_modelo.to_excel(xlsx_path, index=False)
        print(f"Planilha XLSX gerada em: {xlsx_path}")
    except Exception as exc:
        print(
            "Aviso: não foi possível gerar XLSX "
            f"({exc}). O arquivo CSV foi gerado normalmente."
        )

    return df_modelo


def main():
    parser = argparse.ArgumentParser(
        description="Gera e exibe uma planilha com os dados do banco SQLite."
    )
    parser.add_argument("--db", default=DB_NAME, help="Caminho do banco SQLite")
    parser.add_argument("--tabela", default=TABELA_PADRAO, help="Nome da tabela")
    parser.add_argument(
        "--saida",
        default="planilha_presenca",
        help="Nome base do arquivo de saída (sem extensão)",
    )
    args = parser.parse_args()

    try:
        df = gerar_planilha(db_name=args.db, tabela=args.tabela, saida_base=args.saida)
        exibir_planilha_em_janela(df, titulo=f"Planilha - {args.tabela}")
    except Exception as exc:
        print(f"Erro ao gerar planilha: {exc}")


if __name__ == "__main__":
    main()
