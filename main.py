import cv2
import numpy as np
import os
from detector_bancada import BancadaDetector

TINY = True
ARQUIVO_CFG = "yolov4.cfg" if not TINY else "yolov4-tiny.cfg"
ARQUIVO_PESOS = "yolov4.weights" if not TINY else "yolov4-tiny.weights"
ARQUIVO_CLASSES = "coco.names"

# Carregar os nomes das classes
with open(ARQUIVO_CLASSES, "r") as arquivo:
    CLASSES = [linha.strip() for linha in arquivo.readlines()]

# Índice da classe 'gato' na lista COCO
INDICE_GATO = CLASSES.index('gato')

# Defina as coordenadas do retângulo da bancada (x, y, largura, altura)
ROI_BANCADA = (90, 180, 250, 180)  # Ajuste as coordenadas conforme necessário

def carregar_modelo_pretreinado():
    """
    Carrega o modelo pré-treinado YOLO a partir dos arquivos de configuração e pesos.
    Configura o backend e o alvo preferencial para o OpenCV DNN.
    Levanta uma exceção se o modelo não puder ser carregado.
    """
    modelo = cv2.dnn.readNetFromDarknet(ARQUIVO_CFG, ARQUIVO_PESOS)
    modelo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    modelo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    if modelo.empty():
        raise IOError("Não foi possível carregar o modelo de detecção de objetos.")
    return modelo

def preprocessar_frame(frame):
    """
    Pré-processa um frame de imagem para ser usado pelo modelo YOLO.
    Converte a imagem em um blob, normalizando os valores dos pixels e redimensionando para 416x416.
    """
    return cv2.dnn.blobFromImage(frame, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)

def detectar_objetos(frame, modelo):
    """
    Realiza a detecção de objetos em um frame de imagem usando o modelo YOLO.
    Define a entrada do modelo com o frame pré-processado e obtém as camadas de saída.
    Retorna as detecções realizadas pelo modelo.
    """
    modelo.setInput(preprocessar_frame(frame))
    nomes_camadas = modelo.getLayerNames()
    camadas_saida = [nomes_camadas[i - 1] for i in modelo.getUnconnectedOutLayers()]
    return modelo.forward(camadas_saida)

def desenhar_deteccoes(frame, deteccoes, roi_bancada, limiar=0.6, nms_limiar=0.3):
    """
    Desenha as detecções no frame de imagem.
    Filtra as detecções com base no limiar de confiança e aplica Non-Maxima Suppression (NMS).
    Verifica se há um gato na bancada e desenha as caixas delimitadoras.
    """
    (altura, largura) = frame.shape[:2]
    caixas, confiancas, ids_classes = [], [], []
    gato_na_bancada = False

    for saida in deteccoes:
        for deteccao in saida:
            pontuacoes = deteccao[5:]
            id_classe = np.argmax(pontuacoes)
            confianca = pontuacoes[id_classe]
            if confianca > limiar:
                caixa = deteccao[0:4] * np.array([largura, altura, largura, altura])
                (centroX, centroY, largura_caixa, altura_caixa) = caixa.astype("int")
                x = int(centroX - (largura_caixa / 2))
                y = int(centroY - (altura_caixa / 2))
                caixas.append([x, y, int(largura_caixa), int(altura_caixa)])
                confiancas.append(float(confianca))
                ids_classes.append(id_classe)

    indices = cv2.dnn.NMSBoxes(caixas, confiancas, limiar, nms_limiar)

    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (caixas[i][0], caixas[i][1])
            (largura_caixa, altura_caixa) = (caixas[i][2], caixas[i][3])
            if ids_classes[i] == INDICE_GATO:
                centro_gato_x = x + largura_caixa // 2
                centro_gato_y = y + altura_caixa // 2
                if (roi_bancada[0] < centro_gato_x < roi_bancada[0] + roi_bancada[2] and 
                    roi_bancada[1] < centro_gato_y < roi_bancada[1] + roi_bancada[3]):
                    gato_na_bancada = True
                    cor, texto = (0, 0, 255), f"Gato na bancada: {confiancas[i]:.2f}"
                else:
                    cor, texto = (255, 0, 0), f"Gato fora da bancada: {confiancas[i]:.2f}"
                cv2.rectangle(frame, (x, y), (x + largura_caixa, y + altura_caixa), cor, 2)
                cv2.putText(frame, texto, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)

    if gato_na_bancada:
        cv2.putText(frame, "ALERTA!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return gato_na_bancada

def verificar_arquivos():
    arquivos = [ARQUIVO_CFG, ARQUIVO_PESOS, ARQUIVO_CLASSES]
    arquivos_faltando = [arquivo for arquivo in arquivos if not os.path.exists(arquivo)]
    
    print("\nVerificando arquivos necessários:")
    for arquivo in arquivos:
        status = "✓ Arquivo encontrado" if arquivo not in arquivos_faltando else "❌ Arquivo não encontrado"
        print(f"{status}: {arquivo}")
    
    if arquivos_faltando:
        print("\nVocê precisa baixar os seguintes arquivos:")
        for arquivo in arquivos_faltando:
            print(f"- {arquivo}")
        return False
        
    print("\nTodos os arquivos necessários foram encontrados!")
    return True

def main():
    print("\nIniciando verificação de arquivos...")
    if not verificar_arquivos():
        print("Falha na verificação de arquivos")
        return

    print("Inicializando o detector de objetos...")
    try:
        modelo = carregar_modelo_pretreinado()
        print("Modelo carregado com sucesso!")

        caminho_video = "gato.mp4"
        if not os.path.exists(caminho_video):
            print(f"Erro: Arquivo de vídeo não encontrado em: {caminho_video}")
            return
            
        captura_video = cv2.VideoCapture(caminho_video)
        if not captura_video.isOpened():
            print(f"Erro: Não foi possível abrir o vídeo: {caminho_video}")
            return

        print("Vídeo aberto com sucesso!")
        
        # Cria uma instância da classe BancadaDetector, passando as coordenadas da região de interesse (ROI) da bancada.
        # A ROI é definida como um retângulo com as coordenadas (x, y, largura, altura).
        # O detector usará essa ROI para verificar se a bancada está ocupada ou vazia.
        bancada_detector = BancadaDetector(ROI_BANCADA)

        while True:
            ret, frame = captura_video.read()
            if not ret:
                print("Fim do vídeo")
                break

            deteccoes = detectar_objetos(frame, modelo)
            desenhar_deteccoes(frame, deteccoes, ROI_BANCADA)

            img_cinza, img_threshold = bancada_detector.processa_frame(frame)
            bancada_vazia = bancada_detector.verifica_bancada(frame, img_threshold)

            cv2.imshow('Detecta Objetos', frame)
            cv2.imshow('Imagem Cinza', img_cinza)
            cv2.imshow('Imagem Threshold', img_threshold)

            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Erro: {str(e)}")
    finally:
        captura_video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()