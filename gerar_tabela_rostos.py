import sqlite3
import pandas as pd

DB_NAME = 'rostos.db'

def gerar_tabela_rostos():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query('SELECT ord_id, id_rosto, nome, numero_imagem, turma, data_imagem FROM rostos', conn)
    conn.close()
    return df

if __name__ == '__main__':
    tabela = gerar_tabela_rostos()
    if tabela.empty:
        print('Nenhum rosto cadastrado na tabela.')
    else:
        print(tabela.to_string(index=False))
