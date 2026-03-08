import sqlite3

DB_NAME = 'rostos.db'

def atualizar_nome_id(id_rosto, novo_nome):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('UPDATE rostos SET nome = ? WHERE id_rosto = ?', (novo_nome, id_rosto))
    conn.commit()
    conn.close()
    print(f"Nome atualizado para o ID {id_rosto}: {novo_nome}")

if __name__ == "__main__":
    # Exemplo de uso
    id_rosto = input("Digite o ID do rosto: ")
    novo_nome = input("Digite o novo nome: ")
    atualizar_nome_id(id_rosto, novo_nome)
