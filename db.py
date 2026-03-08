import sqlite3

DB_NAME = 'rostos.db'


def _obter_colunas_rostos(cursor):
    cursor.execute('PRAGMA table_info(rostos)')
    return {linha[1] for linha in cursor.fetchall()}


def garantir_coluna_rosto_embeddings(cursor):
    """Padroniza o banco para a coluna única de dados faciais: rosto_embeddings."""
    colunas = _obter_colunas_rostos(cursor)

    if 'rosto_embeddings' not in colunas and 'rosto' in colunas:
        cursor.execute('ALTER TABLE rostos RENAME COLUMN rosto TO rosto_embeddings')
        print("Coluna 'rosto' renomeada para 'rosto_embeddings'.")
        colunas = _obter_colunas_rostos(cursor)

    if 'rosto_embeddings' not in colunas:
        cursor.execute('ALTER TABLE rostos ADD COLUMN rosto_embeddings BLOB')
        print("Coluna 'rosto_embeddings' adicionada à tabela rostos.")


def remover_coluna_embedding_legacy(cursor):
    """Remove a coluna legada 'embedding' quando suportado pela versão do SQLite."""
    colunas = _obter_colunas_rostos(cursor)
    if 'embedding' not in colunas:
        return

    try:
        cursor.execute('ALTER TABLE rostos DROP COLUMN embedding')
        print("Coluna legada 'embedding' removida da tabela rostos.")
    except sqlite3.OperationalError:
        print(
            "Aviso: SQLite desta máquina não suporta DROP COLUMN. "
            "A coluna legada 'embedding' permanecerá no schema, mas não será mais usada."
        )


def garantir_coluna_origem_nome(cursor):
    """Garante compatibilidade com bancos antigos sem a coluna de origem do nome."""
    cursor.execute('PRAGMA table_info(rostos)')
    colunas = {linha[1] for linha in cursor.fetchall()}
    if 'origem_nome' not in colunas:
        cursor.execute("ALTER TABLE rostos ADD COLUMN origem_nome TEXT")
        print("Coluna 'origem_nome' adicionada à tabela rostos.")


def garantir_estrutura_rostos(conn):
    cursor = conn.cursor()
    garantir_coluna_rosto_embeddings(cursor)
    remover_coluna_embedding_legacy(cursor)
    garantir_coluna_origem_nome(cursor)
    conn.commit()

def criar_banco():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS rostos (
            ord_id INTEGER PRIMARY KEY AUTOINCREMENT,
            rosto_embeddings BLOB NOT NULL,
            id_rosto TEXT NOT NULL,
            nome TEXT,
            numero_imagem INTEGER NOT NULL,
            turma TEXT NOT NULL,
            data_imagem TEXT NOT NULL,
            origem_nome TEXT
        )
    ''')
    garantir_estrutura_rostos(conn)
    conn.close()

if __name__ == "__main__":
    criar_banco()
    print("Banco de dados SQLite criado e pronto para uso.")
