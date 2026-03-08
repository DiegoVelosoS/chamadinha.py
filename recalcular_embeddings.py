import argparse
import sqlite3
from typing import Optional

import cv2
import numpy as np


DB_NAME = "rostos.db"


def _decode_rosto_blob(rosto_blob: bytes) -> Optional[np.ndarray]:
    """Decode BLOB bytes into an OpenCV BGR image."""
    if not rosto_blob:
        return None
    img_array = np.frombuffer(rosto_blob, dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)


def _calcular_embedding(rosto_bgr: np.ndarray, model: str):
    """Generate a 128D embedding with face_recognition."""
    import face_recognition

    rosto_rgb = cv2.cvtColor(rosto_bgr, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rosto_rgb, model=model)
    if not encodings:
        return None
    return encodings[0].astype(np.float32)


def _montar_query(apenas_sem_embedding: bool, ord_id: Optional[int], id_rosto: Optional[str]):
    condicoes = []
    params = []

    if apenas_sem_embedding:
        # Sem coluna de embedding persistida, o modo padrão processa tudo.
        pass
    if ord_id is not None:
        condicoes.append("ord_id = ?")
        params.append(ord_id)
    if id_rosto is not None:
        condicoes.append("id_rosto = ?")
        params.append(id_rosto)

    where = f" WHERE {' AND '.join(condicoes)}" if condicoes else ""
    query = f"SELECT ord_id, id_rosto, rosto_embeddings FROM rostos{where} ORDER BY ord_id ASC"
    return query, params


def main():
    parser = argparse.ArgumentParser(
        description="Recalcula embeddings faciais para registros do banco rostos.db"
    )
    parser.add_argument(
        "--db",
        default=DB_NAME,
        help="Caminho do banco SQLite (padrao: rostos.db)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Mantido por compatibilidade; o script sempre processa os registros selecionados",
    )
    parser.add_argument(
        "--model",
        choices=["small", "large"],
        default="small",
        help="Modelo do face_recognition para gerar embedding (padrao: small)",
    )
    parser.add_argument(
        "--ord-id",
        type=int,
        default=None,
        help="Processa apenas o ord_id informado",
    )
    parser.add_argument(
        "--id-rosto",
        type=str,
        default=None,
        help="Processa apenas o id_rosto informado",
    )
    args = parser.parse_args()

    try:
        import face_recognition  # noqa: F401
    except ImportError:
        print("Erro: pacote 'face_recognition' nao esta instalado neste ambiente.")
        print("Instale com:")
        print("  python -m pip install face-recognition")
        return

    conn = sqlite3.connect(args.db)
    cursor = conn.cursor()

    query, params = _montar_query(
        apenas_sem_embedding=False,
        ord_id=args.ord_id,
        id_rosto=args.id_rosto,
    )
    cursor.execute(query, params)
    registros = cursor.fetchall()

    total = len(registros)
    gerados = 0
    sem_rosto = 0
    sem_face = 0

    if total == 0:
        print("Nenhum registro encontrado para processamento.")
        conn.close()
        return

    print(f"Registros selecionados: {total}")
    for ord_id, id_rosto, rosto_blob in registros:

        rosto_bgr = _decode_rosto_blob(rosto_blob)
        if rosto_bgr is None:
            sem_rosto += 1
            print(f"[ord_id={ord_id}] BLOB de rosto invalido; ignorado.")
            continue

        embedding = _calcular_embedding(rosto_bgr, model=args.model)
        if embedding is None:
            sem_face += 1
            print(f"[ord_id={ord_id}] Nao foi possivel gerar embedding (face nao detectada).")
            continue

        gerados += 1
        print(f"[ord_id={ord_id}] embedding gerado em memoria para id_rosto={id_rosto}")

    conn.close()

    print("\nResumo:")
    print(f"- Embeddings gerados: {gerados}")
    print(f"- Sem imagem de rosto valida: {sem_rosto}")
    print(f"- Sem face para encoding: {sem_face}")
    print("- Observacao: os embeddings nao sao mais persistidos no banco.")


if __name__ == "__main__":
    main()
