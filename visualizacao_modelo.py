import cv2
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import os
from modelos import detectar_rostos_opencv, detectar_rostos_dlib, detectar_rostos_mediapipe
from db import garantir_estrutura_rostos

try:
    import face_recognition
except ImportError:
    face_recognition = None

DB_NAME = 'rostos.db'
EMBEDDING_TOLERANCIA = 0.55
EMBEDDING_MARGEM_AMBIGUIDADE = 0.03


def selecionar_imagem_imagens_salvas(pasta_imagens):
    """Lista imagens da pasta e retorna o caminho escolhido pelo usuario."""
    extensoes_validas = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    if not os.path.isdir(pasta_imagens):
        print(f"Pasta nao encontrada: {pasta_imagens}")
        return None

    arquivos = []
    for nome in sorted(os.listdir(pasta_imagens)):
        caminho = os.path.join(pasta_imagens, nome)
        _, ext = os.path.splitext(nome.lower())
        if os.path.isfile(caminho) and ext in extensoes_validas:
            arquivos.append(nome)

    if not arquivos:
        print(f"Nenhuma imagem encontrada em: {pasta_imagens}")
        return None

    print("Imagens disponiveis:")
    for i, nome in enumerate(arquivos, start=1):
        print(f"{i}. {nome}")

    while True:
        escolha = input("Digite o numero da imagem que deseja visualizar: ").strip()
        if not escolha.isdigit():
            print("Entrada invalida. Informe apenas o numero da opcao.")
            continue

        indice = int(escolha)
        if 1 <= indice <= len(arquivos):
            return os.path.join(pasta_imagens, arquivos[indice - 1])

        print(f"Opcao fora do intervalo. Escolha entre 1 e {len(arquivos)}.")

def carregar_rostos_nomeados():
    """Carrega rostos nomeados e calcula embeddings em memória quando possível."""
    try:
        conn = sqlite3.connect(DB_NAME)
        garantir_estrutura_rostos(conn)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT ord_id, nome, rosto_embeddings
            FROM rostos
            WHERE nome IS NOT NULL
              AND TRIM(nome) <> ''
        ''')
        rostos_nomeados = []

        for ord_id, nome, rosto_blob in cursor.fetchall():
            nparr = np.frombuffer(rosto_blob, np.uint8)
            rosto_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if rosto_img is not None:
                embedding = calcular_embedding_rosto(rosto_img) if face_recognition is not None else None

                rostos_nomeados.append({'nome': nome, 'rosto': rosto_img, 'embedding': embedding})

        conn.close()
        return rostos_nomeados
    except Exception as e:
        print(f"Erro ao carregar rostos: {e}")
        return []


def calcular_embedding_rosto(rosto_bgr):
    """Gera embedding facial 128D quando face_recognition esta disponivel."""
    if face_recognition is None:
        return None

    try:
        if rosto_bgr is None or rosto_bgr.size == 0:
            return None

        h, w = rosto_bgr.shape[:2]
        variantes = [rosto_bgr]

        # Faces pequenas costumam falhar no encoder; ampliar melhora a taxa de sucesso.
        if min(h, w) < 160:
            ampliada = cv2.resize(rosto_bgr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            variantes.append(ampliada)

        # Variante com equalizacao para lidar com baixa iluminacao/contraste.
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
    except Exception:
        return None

def comparar_rostos_correlacao(rosto1, rosto2):
    """Compara dois rostos usando correlação normalizada.
    Retorna score entre 0 e 1 (1 = rostos idênticos)."""
    try:
        # Redimensiona ambos para o mesmo tamanho se necessário
        h1, w1 = rosto1.shape[:2]
        h2, w2 = rosto2.shape[:2]
        
        # Usa o menor tamanho para comparação
        min_h = min(h1, h2)
        min_w = min(w1, w2)
        
        if min_h <= 0 or min_w <= 0:
            return 0
        
        rosto1_crop = rosto1[:min_h, :min_w]
        rosto2_crop = rosto2[:min_h, :min_w]
        
        # Converte para escala de cinza
        gray1 = cv2.cvtColor(rosto1_crop, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(rosto2_crop, cv2.COLOR_BGR2GRAY)
        
        # Calcula correlação normalizada
        result = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)
        
        # Normaliza para 0-1
        score = float(np.max(result)) if result.size > 0 else 0
        return min(1.0, max(0.0, score))
        
    except Exception as e:
        return 0


def obter_nome_rosto_por_embedding(rosto_detectado, rostos_nomeados, tolerancia=EMBEDDING_TOLERANCIA):
    """Busca o nome mais proximo via embedding facial."""
    embedding_detectado = calcular_embedding_rosto(rosto_detectado)
    if embedding_detectado is None:
        return None, None

    melhores_por_nome = {}
    for ref in rostos_nomeados:
        ref_emb = ref.get('embedding')
        if ref_emb is None:
            continue
        distancia = float(np.linalg.norm(ref_emb - embedding_detectado))
        nome = ref['nome']
        anterior = melhores_por_nome.get(nome)
        if anterior is None or distancia < anterior:
            melhores_por_nome[nome] = distancia

    if not melhores_por_nome:
        return None, None

    melhores = sorted((dist, nome) for nome, dist in melhores_por_nome.items())
    melhor_distancia, melhor_nome = melhores[0]
    segunda_distancia = melhores[1][0] if len(melhores) > 1 else None

    if segunda_distancia is not None:
        margem = segunda_distancia - melhor_distancia
        if margem < EMBEDDING_MARGEM_AMBIGUIDADE:
            return None, melhor_distancia

    if melhor_distancia <= tolerancia:
        return melhor_nome, melhor_distancia
    return None, melhor_distancia

def obter_nome_rosto(rosto_detectado, rostos_nomeados, threshold=0.45, debug=False):
    """Encontra o nome do rosto: prioriza embedding e usa correlacao como fallback."""
    if not rostos_nomeados:
        if debug:
            print("Nenhum rosto nomeado para comparação")
        return None, 0

    nome_emb, dist_emb = obter_nome_rosto_por_embedding(
        rosto_detectado,
        rostos_nomeados,
        tolerancia=EMBEDDING_TOLERANCIA,
    )

    if debug and dist_emb is not None:
        print(f"Melhor distancia por embedding: {dist_emb:.4f} (tol: {EMBEDDING_TOLERANCIA})")

    if nome_emb:
        # Converte distancia em score aproximado para manter compatibilidade com retorno atual.
        return nome_emb, max(0.0, 1.0 - dist_emb)
    
    melhor_score = 0
    melhor_nome = None
    
    for ref in rostos_nomeados:
        score = comparar_rostos_correlacao(rosto_detectado, ref['rosto'])
        if debug:
            print(f"  {ref['nome']}: {score:.2f}")
        
        if score > melhor_score:
            melhor_score = score
            melhor_nome = ref['nome']
    
    if debug:
        print(f"Melhor score: {melhor_score:.2f}, Threshold: {threshold}")
    
    if melhor_score >= threshold:
        return melhor_nome, melhor_score
    return None, melhor_score

def visualizar_analise_modelo(imagem_path, modelo='opencv'):
    # Carrega a imagem
    img = cv2.imread(imagem_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_display = img_rgb.copy()

    # Carrega rostos nomeados do banco
    rostos_nomeados = carregar_rostos_nomeados()
    total_com_embedding = sum(1 for r in rostos_nomeados if r.get('embedding') is not None)
    print(f"Rostos nomeados carregados: {len(rostos_nomeados)} (com embedding: {total_com_embedding})")

    # Detecta rostos usando o modelo selecionado
    if modelo == 'opencv':
        rostos = detectar_rostos_opencv(img)
    elif modelo == 'dlib':
        rostos = detectar_rostos_dlib(img)
    elif modelo == 'mediapipe':
        rostos = detectar_rostos_mediapipe(img)
    else:
        raise ValueError('Modelo desconhecido')

    print(f"Rostos detectados: {len(rostos)}")

    # Estilo das caixas, numeração e legenda
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    text_color = (0, 0, 0)
    bg_color = (57, 255, 20)

    altura_img = img_display.shape[0]
    largura_img = img_display.shape[1]
    legenda_largura = 300
    legenda_bg = 235

    # Monta lista de itens reconhecidos
    itens_temp = []
    for (x, y, w, h) in rostos:
        rosto = img[y:y+h, x:x+w]
        nome, _ = obter_nome_rosto(rosto, rostos_nomeados, threshold=0.45, debug=False)
        label = nome if nome else "Sem nome"
        itens_temp.append({
            'nome': label,
            'face': (int(x), int(y), int(w), int(h)),
            'face_center_x': int(x + (w // 2))
        })

    # Ordena da esquerda para a direita e então numera
    itens_temp.sort(key=lambda item: item['face_center_x'])
    itens_legenda = []
    for idx, item in enumerate(itens_temp, start=1):
        item['numero'] = idx
        itens_legenda.append(item)

    # Cria canvas com área de legenda à direita
    img_saida = np.full((altura_img, largura_img + legenda_largura, 3), legenda_bg, dtype=np.uint8)
    img_saida[:, :largura_img] = img_display

    # Desenha boxes e numeração próxima de cada rosto
    for item in itens_legenda:
        numero = item['numero']
        x, y, w, h = item['face']

        cv2.rectangle(img_saida, (x, y), (x + w, y + h), bg_color, 2)

        badge_text = str(numero)
        text_w, text_h = cv2.getTextSize(badge_text, font, 0.55, 2)[0]
        badge_x1 = max(0, x - 2)
        badge_y1 = max(0, y - text_h - 10)
        badge_x2 = badge_x1 + text_w + 10
        badge_y2 = badge_y1 + text_h + 8

        cv2.rectangle(img_saida, (badge_x1, badge_y1), (badge_x2, badge_y2), bg_color, -1)
        cv2.putText(img_saida, badge_text, (badge_x1 + 5, badge_y2 - 4), font, 0.55, text_color, 2)

    # Desenha título e itens da legenda
    legenda_x = largura_img + 14
    y_cursor = 28
    cv2.putText(img_saida, 'Legenda', (legenda_x, y_cursor), font, 0.8, (20, 20, 20), 2)
    y_cursor += 26

    line_h = 24
    for item in itens_legenda:
        texto = f"{item['numero']:>2} - {item['nome']}"
        cv2.putText(img_saida, texto, (legenda_x, y_cursor), font, 0.55, (20, 20, 20), 1)
        y_cursor += line_h
        if y_cursor > altura_img - 10:
            break

    plt.figure(figsize=(12, 8))
    plt.imshow(img_saida)
    plt.title(f'Rostos identificados ({modelo})')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    pasta_imagens = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'imagens_salvas')
    imagem_escolhida = selecionar_imagem_imagens_salvas(pasta_imagens)

    if imagem_escolhida:
        visualizar_analise_modelo(imagem_escolhida, modelo='opencv')
