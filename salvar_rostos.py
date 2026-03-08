
import os
import cv2
import numpy as np
import sqlite3
from datetime import datetime
from modelos import detectar_rostos_opencv
from db import garantir_estrutura_rostos
import matplotlib.pyplot as plt

try:
    import face_recognition
except ImportError:
    face_recognition = None

PASTA_IMAGENS = 'imagens_salvas'
PASTA_ABS = os.path.join(os.path.dirname(__file__), PASTA_IMAGENS)
os.makedirs(PASTA_ABS, exist_ok=True)

DB_NAME = 'rostos.db'
EMBEDDING_TOLERANCIA = 0.55
EMBEDDING_MARGEM_AMBIGUIDADE = 0.03


def _blob_para_imagem(rosto_blob):
    if rosto_blob is None:
        return None
    arr = np.frombuffer(rosto_blob, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def exibir_comparacao_reconhecimento(rosto_novo, rosto_referencia, nome_reconhecido, id_novo, distancia):
    """Exibe lado a lado o rosto novo detectado e um rosto de referência do banco."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"Reconhecimento automático: '{nome_reconhecido}' (distância: {distancia:.3f})", fontsize=14)

    if rosto_novo is not None:
        axes[0].imshow(cv2.cvtColor(rosto_novo, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Rosto Detectado Agora\nid={id_novo}")
    axes[0].axis("off")

    if rosto_referencia is not None:
        axes[1].imshow(cv2.cvtColor(rosto_referencia, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"Rosto Cadastrado ({nome_reconhecido})\nna Base de Dados")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


def resposta_sim_nao(pergunta):
    """Pergunta ao usuário e retorna True para sim ou False para não."""
    while True:
        resp = input(pergunta).strip().lower()
        if resp in {"s", "sim", "y", "yes"}:
            return True
        if resp in {"n", "nao", "não", "no"}:
            return False
        print("Resposta inválida. Digite s/sim ou n/nao.")


def buscar_rosto_referencia(cursor, nome_reconhecido):
    """Busca o rosto mais antigo com o nome reconhecido para usar como referência."""
    cursor.execute(
        """
        SELECT rosto_embeddings
        FROM rostos
        WHERE LOWER(TRIM(nome)) = LOWER(TRIM(?))
        ORDER BY ord_id ASC
        LIMIT 1
        """,
        (nome_reconhecido,),
    )
    resultado = cursor.fetchone()
    if resultado:
        return _blob_para_imagem(resultado[0])
    return None


def carregar_referencias_nomeadas(cursor):
    """Carrega embeddings já nomeados para reconhecimento automático."""
    cursor.execute('''
        SELECT nome, rosto_embeddings
        FROM rostos
        WHERE nome IS NOT NULL
          AND TRIM(nome) <> ''
          AND rosto_embeddings IS NOT NULL
    ''')
    referencias = []
    for nome, rosto_blob in cursor.fetchall():
        rosto_img = _blob_para_imagem(rosto_blob)
        embedding = calcular_embedding_rosto(rosto_img)
        if embedding is not None and embedding.size == 128:
            referencias.append({'nome': nome, 'embedding': embedding})
    return referencias


def calcular_embedding_rosto(rosto_bgr):
    """Gera embedding facial (128D) para comparação entre rostos."""
    if face_recognition is None:
        return None

    if rosto_bgr is None or rosto_bgr.size == 0:
        return None

    h, w = rosto_bgr.shape[:2]
    variantes = [rosto_bgr]

    if min(h, w) < 160:
        ampliada = cv2.resize(rosto_bgr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        variantes.append(ampliada)

    ycrcb = cv2.cvtColor(rosto_bgr, cv2.COLOR_BGR2YCrCb)
    y_channel, cr_channel, cb_channel = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y_channel)
    eq_bgr = cv2.cvtColor(cv2.merge([y_eq, cr_channel, cb_channel]), cv2.COLOR_YCrCb2BGR)
    variantes.append(eq_bgr)

    for variante in variantes:
        rosto_rgb = cv2.cvtColor(variante, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rosto_rgb, model='small')
        if encodings:
            return encodings[0].astype(np.float32)

    return None


def reconhecer_nome_por_embedding(embedding, referencias, tolerancia=EMBEDDING_TOLERANCIA):
    """Retorna o nome mais próximo quando a distância estiver abaixo da tolerância."""
    if embedding is None or not referencias:
        return None, None

    melhores_por_nome = {}
    for ref in referencias:
        distancia = float(np.linalg.norm(ref['embedding'] - embedding))
        nome = ref['nome']
        atual = melhores_por_nome.get(nome)
        if atual is None or distancia < atual:
            melhores_por_nome[nome] = distancia

    ranking = sorted((dist, nome) for nome, dist in melhores_por_nome.items())
    menor_distancia, melhor_nome = ranking[0]
    segunda_distancia = ranking[1][0] if len(ranking) > 1 else None

    if segunda_distancia is not None:
        margem = segunda_distancia - menor_distancia
        if margem < EMBEDDING_MARGEM_AMBIGUIDADE:
            return None, menor_distancia

    if menor_distancia <= tolerancia:
        return melhor_nome, menor_distancia
    return None, menor_distancia

def salvar_rostos(imagem_path, turma):
    imagem = cv2.imread(imagem_path)
    if imagem is None:
        print('Imagem inválida ou não encontrada.')
        return

    rostos = detectar_rostos_opencv(imagem)
    conn = sqlite3.connect(DB_NAME)
    garantir_estrutura_rostos(conn)
    cursor = conn.cursor()
    referencias_nomeadas = carregar_referencias_nomeadas(cursor)

    if face_recognition is None:
        print("Aviso: pacote 'face_recognition' não está instalado. O reconhecimento automático foi desativado.")

    reconhecidos = 0
    reconhecidos_confirmados = 0
    reconhecidos_rejeitados = 0
    novos = 0
    sem_embedding = 0

    numero_imagem = int(os.path.splitext(os.path.basename(imagem_path))[0].replace('img','').replace('.jpg',''))
    data_imagem = input('Data e hora da imagem (AAAA-MM-DD HH:MM): ')
    
    # Primeiro, exibe todos os rostos detectados e reconhecidos
    print(f"\n{len(rostos)} rosto(s) detectado(s) na imagem.")
    rostos_com_reconhecimento = []
    
    for idx, (x, y, w, h) in enumerate(rostos):
        rosto = imagem[y:y+h, x:x+w]
        embedding = calcular_embedding_rosto(rosto)
        nome_reconhecido, distancia = reconhecer_nome_por_embedding(embedding, referencias_nomeadas)
        
        ord_id_str = f'{numero_imagem:03d}'
        idx_str = f'{idx:03d}'
        dt = datetime.strptime(data_imagem, '%Y-%m-%d %H:%M')
        data_str = dt.strftime('%Y%m%d%H%M')
        id_rosto = f'{ord_id_str}-{idx_str}-{data_str}'
        
        if embedding is None:
            sem_embedding += 1
        
        if nome_reconhecido:
            reconhecidos += 1
            print(f"  - Rosto {idx+1}: Reconhecido como '{nome_reconhecido}' (distância: {distancia:.3f})")
            rostos_com_reconhecimento.append((idx, rosto, nome_reconhecido, distancia, id_rosto, x, y, w, h))
        else:
            print(f"  - Rosto {idx+1}: Não reconhecido (novo)")
    
    # Mensagem informativa
    if reconhecidos > 0:
        print(f"\n{reconhecidos} rosto(s) reconhecido(s) automaticamente.")
        print("Iniciando validação manual para confirmar identidades...\n")
    
    # Agora processa cada rosto
    for idx, (x, y, w, h) in enumerate(rostos):
        rosto = imagem[y:y+h, x:x+w]
        _, buffer = cv2.imencode('.jpg', rosto)
        embedding = calcular_embedding_rosto(rosto)
        nome_reconhecido, distancia = reconhecer_nome_por_embedding(embedding, referencias_nomeadas)

        ord_id_str = f'{numero_imagem:03d}'
        idx_str = f'{idx:03d}'
        dt = datetime.strptime(data_imagem, '%Y-%m-%d %H:%M')
        data_str = dt.strftime('%Y%m%d%H%M')
        id_rosto = f'{ord_id_str}-{idx_str}-{data_str}'

        # Se foi reconhecido, precisa validar com o usuário
        if nome_reconhecido:
            print(f"\n--- Validando Reconhecimento do Rosto {idx+1} ---")
            print(f"ID: {id_rosto}")
            print(f"Reconhecido como: '{nome_reconhecido}' (distância: {distancia:.3f})")
            
            # Busca um rosto de referência do banco
            rosto_referencia = buscar_rosto_referencia(cursor, nome_reconhecido)
            
            if rosto_referencia is not None:
                # Exibe os dois rostos para comparação
                print("Exibindo comparação visual...")
                exibir_comparacao_reconhecimento(rosto, rosto_referencia, nome_reconhecido, id_rosto, distancia)
                
                # Pergunta ao usuário
                mesma_pessoa = resposta_sim_nao(
                    f"Após fechar a janela, o rosto detectado é a mesma pessoa de '{nome_reconhecido}'? (s/n): "
                )
                
                if mesma_pessoa:
                    # Usuário confirmou: salva com o nome reconhecido
                    reconhecidos_confirmados += 1
                    origem_nome = 'automatico'
                    print(f"✓ Reconhecimento confirmado: '{nome_reconhecido}'")
                else:
                    # Usuário rejeitou: salva sem nome
                    reconhecidos_rejeitados += 1
                    nome_reconhecido = None
                    origem_nome = None
                    novos += 1
                    print(f"✗ Reconhecimento rejeitado. Rosto salvo sem nome (novo ID criado).")
            else:
                # Não encontrou referência (caso raro), aceita automaticamente
                reconhecidos_confirmados += 1
                origem_nome = 'automatico'
                print(f"✓ Sem referência anterior. Reconhecimento aceito automaticamente.")
        else:
            # Não foi reconhecido: salva sem nome
            novos += 1
            origem_nome = None
            print(f"\nRosto {idx+1} (ID: {id_rosto}): Salvo como novo registro sem nome.")

        # Insere no banco de dados
        cursor.execute(
            '''INSERT INTO rostos (rosto_embeddings, id_rosto, nome, numero_imagem, turma, data_imagem, origem_nome)
               VALUES (?, ?, ?, ?, ?, ?, ?)''',
            (
                buffer.tobytes(),
                id_rosto,
                nome_reconhecido,
                numero_imagem,
                turma,
                data_imagem,
                origem_nome,
            )
        )

    conn.commit()
    conn.close()
    print(f'\n{"="*60}')
    print(f'Processamento concluído: {len(rostos)} rosto(s) processado(s).')
    print(f'{"="*60}')
    print(f'- Reconhecidos automaticamente: {reconhecidos}')
    if reconhecidos > 0:
        print(f'  • Confirmados pelo usuário: {reconhecidos_confirmados}')
        print(f'  • Rejeitados pelo usuário: {reconhecidos_rejeitados}')
    print(f'- Novos/sem correspondência: {novos}')
    if sem_embedding > 0:
        print(f'- Sem embedding válido: {sem_embedding}')
    print(f'{"="*60}\n')

def upload_imagem():
    caminho = input('Caminho do arquivo de imagem para upload: ')
    arquivos = [f for f in os.listdir(PASTA_ABS) if f.startswith('img') and f.endswith('.jpg')]
    if arquivos:
        nums = [int(f[3:-4]) for f in arquivos if f[3:-4].isdigit()]
        numero = max(nums) + 1 if nums else 1
    else:
        numero = 1
    nome_arquivo = f'img{numero}.jpg'
    destino = os.path.join(PASTA_ABS, nome_arquivo)
    img = cv2.imread(caminho)
    if img is None:
        print('Imagem inválida ou não encontrada.')
        return None
    cv2.imwrite(destino, img)
    print(f'Imagem salva em: {destino}')
    return destino

if __name__ == '__main__':
    imagem_path = upload_imagem()
    if imagem_path:
        turma = input('Turma da imagem: ')
        salvar_rostos(imagem_path, turma)
