import argparse
import math
import sqlite3

import cv2
import matplotlib.pyplot as plt
import numpy as np

from db import garantir_estrutura_rostos

DB_NAME = "rostos.db"
COLUNAS = "ord_id, rosto_embeddings, id_rosto, nome"


def _decodificar_rosto(rosto_blob):
    """Converte o BLOB do banco em imagem BGR (OpenCV)."""
    if not rosto_blob:
        return None

    img_array = np.frombuffer(rosto_blob, dtype=np.uint8)
    if img_array.size == 0:
        return None

    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)


def _buscar_rostos(nome=None, id_rosto=None):
    conn = sqlite3.connect(DB_NAME)
    try:
        garantir_estrutura_rostos(conn)
        cursor = conn.cursor()
        query = f"SELECT {COLUNAS} FROM rostos"
        filtros = []
        valores = []

        if nome:
            filtros.append("nome LIKE ?")
            valores.append(f"%{nome}%")

        if id_rosto:
            filtros.append("id_rosto = ?")
            valores.append(id_rosto)

        if filtros:
            query += " WHERE " + " AND ".join(filtros)

        query += " ORDER BY ord_id"
        cursor.execute(query, valores)
        return cursor.fetchall()
    finally:
        conn.close()


def exibir_todos_os_rostos(nome=None, id_rosto=None):
    registros = _buscar_rostos(nome=nome, id_rosto=id_rosto)
    if not registros:
        print("Nenhum rosto encontrado para os filtros informados.")
        return

    total = len(registros)
    colunas = min(4, total)
    linhas = math.ceil(total / colunas)

    fig, eixos = plt.subplots(linhas, colunas, figsize=(4 * colunas, 4 * linhas))
    eixos = np.atleast_1d(eixos).ravel()

    for eixo in eixos:
        eixo.axis("off")

    for idx, (ord_id, rosto_blob, id_rosto, nome) in enumerate(registros):
        eixo = eixos[idx]
        img = _decodificar_rosto(rosto_blob)

        if img is None:
            eixo.text(
                0.5,
                0.5,
                "Imagem invalida",
                ha="center",
                va="center",
                fontsize=10,
            )
            titulo = f"Nome: {nome or 'Sem nome'} | ID: {id_rosto}"
            eixo.set_title(titulo, fontsize=9)
            continue

        eixo.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        titulo = f"Nome: {nome or 'Sem nome'} | ID: {id_rosto}"
        eixo.set_title(titulo, fontsize=9)

    plt.tight_layout()
    plt.show()


def _ler_argumentos():
    parser = argparse.ArgumentParser(
        description="Exibe rostos cadastrados com nome e ID, com filtros opcionais."
    )
    parser.add_argument(
        "--nome",
        type=str,
        help="Filtra por nome (busca parcial, sem diferenciar maiusculas/minusculas).",
    )
    parser.add_argument(
        "--id",
        dest="id_rosto",
        type=str,
        help="Filtra por ID exato do rosto.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _ler_argumentos()
    exibir_todos_os_rostos(nome=args.nome, id_rosto=args.id_rosto)
