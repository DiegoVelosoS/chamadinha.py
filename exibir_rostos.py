import sqlite3
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from db import garantir_estrutura_rostos

DB_NAME = 'rostos.db'

# Pasta para salvar imagens temporárias
PASTA_IMAGENS = 'imagens_salvas'
os.makedirs(PASTA_IMAGENS, exist_ok=True)

def exibir_rostos_salvos():
    conn = sqlite3.connect(DB_NAME)
    garantir_estrutura_rostos(conn)
    cursor = conn.cursor()
    cursor.execute('SELECT ord_id, rosto_embeddings, id_rosto, nome, numero_imagem, turma, data_imagem FROM rostos')
    rostos = cursor.fetchall()
    conn.close()
    for idx, (id_, rosto_blob, id_rosto, nome, numero_imagem, turma, data_imagem) in enumerate(rostos):
        img_array = np.frombuffer(rostos[idx][1], dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"ID: {id_rosto} | Nome: {nome} | Turma: {turma} | Data: {data_imagem}")
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    exibir_rostos_salvos()
