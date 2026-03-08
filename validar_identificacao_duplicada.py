import sqlite3
from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from db import garantir_estrutura_rostos


DB_NAME = "rostos.db"


def blob_para_imagem(rosto_blob: bytes) -> Optional[np.ndarray]:
    if not rosto_blob:
        return None
    arr = np.frombuffer(rosto_blob, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def exibir_comparacao(
    nome: str,
    registro_novo: Tuple,
    registro_antigo: Tuple,
) -> None:
    ord_novo, id_novo, _, rosto_blob_novo, _, _, _ = registro_novo
    ord_antigo, id_antigo, _, rosto_blob_antigo, _, _, _ = registro_antigo

    img_nova = blob_para_imagem(rosto_blob_novo)
    img_antiga = blob_para_imagem(rosto_blob_antigo)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f"Comparacao para nome: {nome}", fontsize=12)

    if img_antiga is not None:
        axes[0].imshow(cv2.cvtColor(img_antiga, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Mais antigo\nord_id={ord_antigo}\nid={id_antigo}")
    axes[0].axis("off")

    if img_nova is not None:
        axes[1].imshow(cv2.cvtColor(img_nova, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"Mais recente\nord_id={ord_novo}\nid={id_novo}")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


def resposta_sim_nao(pergunta: str) -> bool:
    while True:
        resp = input(pergunta).strip().lower()
        if resp in {"s", "sim", "y", "yes"}:
            return True
        if resp in {"n", "nao", "não", "no"}:
            return False
        print("Resposta invalida. Digite s/sim ou n/nao.")


def buscar_primeiro_sem_nome(cursor: sqlite3.Cursor):
    cursor.execute(
        """
        SELECT ord_id, id_rosto, nome, rosto_embeddings, numero_imagem, turma, data_imagem
        FROM rostos
        WHERE nome IS NULL OR TRIM(nome) = ''
        ORDER BY ord_id ASC
        LIMIT 1
        """
    )
    return cursor.fetchone()


def buscar_mais_antigo_mesmo_nome(cursor: sqlite3.Cursor, nome: str, ord_id_atual: int):
    cursor.execute(
        """
                SELECT ord_id, id_rosto, nome, rosto_embeddings, numero_imagem, turma, data_imagem
        FROM rostos
        WHERE LOWER(TRIM(nome)) = LOWER(TRIM(?))
          AND ord_id <> ?
        ORDER BY ord_id ASC
        LIMIT 1
        """,
        (nome, ord_id_atual),
    )
    return cursor.fetchone()


def buscar_nomes_duplicados(cursor: sqlite3.Cursor):
    cursor.execute(
        """
        SELECT TRIM(nome) AS nome_limpo, COUNT(*) AS total
        FROM rostos
        WHERE nome IS NOT NULL
          AND TRIM(nome) <> ''
        GROUP BY LOWER(TRIM(nome))
        HAVING COUNT(*) > 1
        ORDER BY nome_limpo ASC
        """
    )
    return cursor.fetchall()


def buscar_registros_por_nome(cursor: sqlite3.Cursor, nome: str):
    cursor.execute(
        """
        SELECT ord_id, id_rosto, nome, rosto_embeddings, numero_imagem, turma, data_imagem
        FROM rostos
        WHERE LOWER(TRIM(nome)) = LOWER(TRIM(?))
        ORDER BY ord_id ASC
        """,
        (nome,),
    )
    return cursor.fetchall()


def atualizar_nome(
    cursor: sqlite3.Cursor,
    ord_id: int,
    nome: str,
) -> None:
    cursor.execute(
        "UPDATE rostos SET nome = ? WHERE ord_id = ?",
        (nome, ord_id),
    )


def processar_registro(cursor: sqlite3.Cursor, registro_novo: Tuple) -> None:
    ord_novo, id_novo, _, rosto_blob_novo, _, _, _ = registro_novo

    while True:
        nome_informado = input(f"Digite o nome para o rosto ord_id={ord_novo} (id={id_novo}): ").strip()
        if nome_informado:
            break
        print("Nome vazio. Tente novamente.")

    while True:
        registro_antigo = buscar_mais_antigo_mesmo_nome(cursor, nome_informado, ord_novo)

        if registro_antigo is None:
            atualizar_nome(cursor, ord_novo, nome_informado)
            print(f"Nome '{nome_informado}' salvo para ord_id={ord_novo}.")
            return

        exibir_comparacao(nome_informado, registro_novo, registro_antigo)
        mesma_pessoa = resposta_sim_nao("Apos fechar a janela, e a mesma pessoa? (s/n): ")

        ord_antigo, id_antigo, _, _, _, _, _ = registro_antigo

        if mesma_pessoa:
            cursor.execute(
                "UPDATE rostos SET nome = ? WHERE ord_id = ?",
                (nome_informado, ord_antigo),
            )
            cursor.execute("DELETE FROM rostos WHERE ord_id = ?", (ord_novo,))

            print(
                f"Duplicidade confirmada: mantido ord_id={ord_antigo} (id={id_antigo}) "
                f"e excluido ord_id={ord_novo} (mais recente)."
            )
            return

        nome_informado = input("Nao e a mesma pessoa. Digite um novo nome: ").strip()
        while not nome_informado:
            nome_informado = input("Nome vazio. Digite um novo nome: ").strip()


def validar_nomes_ja_cadastrados(cursor: sqlite3.Cursor) -> None:
    while True:
        duplicados = buscar_nomes_duplicados(cursor)
        if not duplicados:
            print("Nao ha nomes duplicados para validar.")
            return

        nome, total = duplicados[0]
        registros = buscar_registros_por_nome(cursor, nome)
        if len(registros) < 2:
            continue

        registro_antigo = registros[0]
        registro_novo = registros[-1]

        ord_antigo, id_antigo, nome_antigo, _, _, _, _ = registro_antigo
        ord_novo, id_novo, nome_novo, rosto_blob_novo, _, _, _ = registro_novo

        print(
            f"\nNome duplicado detectado: '{nome}' (total: {total}). "
            f"Comparando ord_id antigo={ord_antigo} com recente={ord_novo}."
        )
        exibir_comparacao(nome, registro_novo, registro_antigo)

        mesma_pessoa = resposta_sim_nao("Apos fechar a janela, e a mesma pessoa? (s/n): ")

        if mesma_pessoa:
            nome_final = nome_antigo if nome_antigo else nome_novo

            cursor.execute(
                "UPDATE rostos SET nome = ? WHERE ord_id = ?",
                (nome_final, ord_antigo),
            )
            cursor.execute("DELETE FROM rostos WHERE ord_id = ?", (ord_novo,))

            print(
                f"Duplicidade confirmada: mantido ord_id={ord_antigo} (id={id_antigo}) "
                f"e excluido ord_id={ord_novo} (id={id_novo})."
            )
        else:
            novo_nome = input(
                f"Nao e a mesma pessoa. Digite um novo nome para ord_id={ord_novo} (id={id_novo}): "
            ).strip()
            while not novo_nome:
                novo_nome = input("Nome vazio. Digite um novo nome: ").strip()

            atualizar_nome(cursor, ord_novo, novo_nome)
            print(f"ord_id={ord_novo} atualizado para o novo nome '{novo_nome}'.")


def main():
    conn = sqlite3.connect(DB_NAME)
    garantir_estrutura_rostos(conn)
    cursor = conn.cursor()

    print("\n--- Validacao de Nomes Duplicados ---")
    print("Este script processa rostos sem nome e valida duplicidade visualmente.\n")

    while True:
        registro = buscar_primeiro_sem_nome(cursor)
        if registro is None:
            print("Nao ha rostos sem nome para processar.")
            break

        processar_registro(cursor, registro)
        conn.commit()

        continuar = resposta_sim_nao("Deseja processar o proximo rosto sem nome? (s/n): ")
        if not continuar:
            break

    print("\nIniciando validacao de nomes ja cadastrados e repetidos...")
    validar_nomes_ja_cadastrados(cursor)
    conn.commit()

    conn.close()
    print("Processamento concluido.")


if __name__ == "__main__":
    main()
