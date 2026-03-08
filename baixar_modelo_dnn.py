import urllib.request
import os

print("Baixando modelo DNN do OpenCV (mais preciso)...\n")

# URLs dos arquivos
files = {
    'deploy.prototxt': 'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt',
    'res10_300x300_ssd_iter_140000.caffemodel': 'https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel'
}

for filename, url in files.items():
    if os.path.exists(filename):
        print(f"✓ {filename} já existe")
    else:
        print(f"Baixando {filename}...")
        try:
            urllib.request.urlretrieve(url, filename)
            print(f"✓ {filename} baixado com sucesso!")
        except Exception as e:
            print(f"✗ Erro ao baixar {filename}: {e}")

print("\n" + "="*50)
print("Download concluído!")
print("="*50)
print("\nAgora execute: python testar_deteccao.py")
