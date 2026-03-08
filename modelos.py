import cv2
from PIL import Image
import numpy as np
import os
import urllib.request


def _garantir_modelo_dnn(config_file, model_file):
    """Baixa os arquivos do detector DNN se não estiverem disponíveis localmente."""
    if os.path.exists(config_file) and os.path.exists(model_file):
        return

    print("Baixando modelo DNN de fallback...")
    base_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector"
    try:
        if not os.path.exists(config_file):
            urllib.request.urlretrieve(f"{base_url}/deploy.prototxt", config_file)
        if not os.path.exists(model_file):
            urllib.request.urlretrieve(
                f"{base_url}/res10_300x300_ssd_iter_140000.caffemodel",
                model_file,
            )
        print("✓ Modelo DNN baixado com sucesso!")
    except Exception as exc:
        print(f"Aviso: não foi possível baixar o modelo DNN ({exc}).")

# ============================================================================
# DETECTOR YUNET (MELHOR OPÇÃO - OpenCV 2023, state-of-the-art)
# ============================================================================

class YuNetDetector:
    """Detector YuNet - modelo moderno do OpenCV (2023) - alta precisão"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(YuNetDetector, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.model_path = "face_detection_yunet_2023mar.onnx"
        self._download_model()
        self.detector = cv2.FaceDetectorYN.create(
            self.model_path,
            "",
            (320, 320),
            0.6,  # score threshold
            0.3,  # nms threshold
            5000
        )
        self._initialized = True
    
    def _download_model(self):
        """Baixa o modelo se não existir"""
        if not os.path.exists(self.model_path):
            print("Baixando modelo YuNet (detecção avançada)...")
            url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
            urllib.request.urlretrieve(url, self.model_path)
            print("✓ Modelo YuNet baixado com sucesso!")
    
    def detectar(self, imagem):
        """Detecta rostos na imagem"""
        height, width = imagem.shape[:2]
        self.detector.setInputSize((width, height))
        
        _, faces = self.detector.detect(imagem)
        
        if faces is None:
            return np.array([])
        
        # Converter formato para (x, y, width, height)
        rostos = []
        for face in faces:
            x, y, w, h = face[:4].astype(int)
            # Garantir que está dentro da imagem
            x = max(0, x)
            y = max(0, y)
            w = min(w, width - x)
            h = min(h, height - y)
            rostos.append((x, y, w, h))
        
        return np.array(rostos)

def detectar_rostos_yunet(imagem):
    """Detector YuNet - RECOMENDADO (melhor precisão)"""
    try:
        detector = YuNetDetector()
        return detector.detectar(imagem)
    except Exception as e:
        print(f"Aviso: YuNet falhou ({e}), usando detector DNN...")
        return detectar_rostos_opencv_dnn(imagem)

# ============================================================================
# DETECTORES ALTERNATIVOS (FALLBACK)
# ============================================================================

# Modelo OpenCV DNN (mais preciso)
def detectar_rostos_opencv_dnn(imagem, confidence_threshold=0.20):
    """Detector baseado em DNN - muito mais preciso que Haar Cascade"""
    # Carregar modelo pré-treinado
    model_file = "res10_300x300_ssd_iter_140000.caffemodel"
    config_file = "deploy.prototxt"
    
    _garantir_modelo_dnn(config_file, model_file)

    try:
        net = cv2.dnn.readNetFromCaffe(config_file, model_file)
    except:
        # Se não tiver os arquivos, usa Haar Cascade como fallback
        print("Aviso: Arquivos DNN não encontrados. Usando Haar Cascade...")
        return detectar_rostos_opencv_haar(imagem)
    
    (h, w) = imagem.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(imagem, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    
    net.setInput(blob)
    detections = net.forward()
    
    rostos_brutos = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x2, y2) = box.astype("int")
            
            # Proteger contra coordenadas negativas ou fora da imagem
            x = max(0, x)
            y = max(0, y)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            width = x2 - x
            height = y2 - y
            
            # Filtro 1: Tamanho mínimo (evita números pequenos)
            if width < 30 or height < 30:
                continue
            
            # Filtro 2: Tamanho máximo relativo (evita detecções muito grandes)
            if width > w * 0.5 or height > h * 0.5:
                continue
            
            # Filtro 3: Razão de aspecto (rostos são aproximadamente quadrados)
            aspect_ratio = width / float(height)
            if aspect_ratio < 0.6 or aspect_ratio > 1.6:
                continue
            
            rostos_brutos.append((x, y, width, height, confidence))
    
    # Aplicar Non-Maximum Suppression para eliminar sobreposições
    rostos = non_max_suppression(rostos_brutos)
    
    return rostos

def non_max_suppression(rostos, overlap_thresh=0.3):
    """Remove detecções sobrepostas mantendo as de maior confiança"""
    if len(rostos) == 0:
        return np.array([])
    
    # Converter para array numpy
    boxes = np.array([(x, y, x+w, y+h, conf) for x, y, w, h, conf in rostos])
    
    # Separar coordenadas e confiança
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    confs = boxes[:, 4]
    
    # Calcular área de cada box
    areas = (x2 - x1) * (y2 - y1)
    
    # Ordenar por confiança (maior primeiro)
    idxs = np.argsort(confs)[::-1]
    
    pick = []
    while len(idxs) > 0:
        # Pegar o box com maior confiança
        i = idxs[0]
        pick.append(i)
        
        # Calcular IoU com os demais boxes
        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        
        overlap = (w * h) / areas[idxs[1:]]
        
        # Remover boxes com IoU maior que o threshold
        idxs = np.delete(idxs, np.concatenate(([0], np.where(overlap > overlap_thresh)[0] + 1)))
    
    # Retornar no formato (x, y, width, height)
    resultado = []
    for i in pick:
        x, y, x2, y2, _ = boxes[i]
        resultado.append((int(x), int(y), int(x2-x), int(y2-y)))
    
    return np.array(resultado)
    return np.array(rostos)

# Modelo OpenCV Haar Cascade (fallback)
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade_opencv = cv2.CascadeClassifier(CASCADE_PATH)

def detectar_rostos_opencv_haar(imagem):
    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    # Parâmetros balanceados
    rostos = face_cascade_opencv.detectMultiScale(
        gray, 
        scaleFactor=1.08,    # Meio termo
        minNeighbors=4,      # Meio termo entre 3 e 5
        minSize=(25, 25),    # Meio termo
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return rostos

def detectar_rostos_opencv(imagem):
    """
    Detector principal - usa a melhor opção disponível:
    1º YuNet (2023) - state-of-the-art
    2º DNN (fallback)
    3º Haar Cascade (fallback final)
    """
    return detectar_rostos_yunet(imagem)

# Modelo PIL + dlib (alternativo leve)
try:
    import dlib
    detector_dlib = dlib.get_frontal_face_detector()
    def detectar_rostos_dlib(imagem):
        # Converte para RGB se necessário
        if len(imagem.shape) == 3:
            img_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = imagem
        dets = detector_dlib(img_rgb, 1)
        rostos = [(d.left(), d.top(), d.width(), d.height()) for d in dets]
        return rostos
except ImportError:
    detector_dlib = None
    def detectar_rostos_dlib(imagem):
        raise ImportError('dlib não está instalado')

# Modelo PIL + mediapipe (alternativo leve)
try:
    import mediapipe as mp
    mp_face_detection = mp.solutions.face_detection
    def detectar_rostos_mediapipe(imagem):
        with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
            img_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
            results = face_detection.process(img_rgb)
            rostos = []
            if results.detections:
                for det in results.detections:
                    bbox = det.location_data.relative_bounding_box
                    h, w, _ = img_rgb.shape
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    rostos.append((x, y, width, height))
            return rostos
except (ImportError, AttributeError):
    mp_face_detection = None
    def detectar_rostos_mediapipe(imagem):
        raise ImportError('mediapipe não está instalado corretamente')
