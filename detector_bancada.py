import cv2
import numpy as np

class BancadaDetector:
    def __init__(self, roi_bancada, limite_ocupada=1500):
        # Inicializa a classe BancadaDetector com a região de interesse (ROI) da bancada e o limite de ocupação.
        # A ROI é definida como um retângulo com as coordenadas (x, y, largura, altura).
        self.roi_bancada = roi_bancada
        self.limite_ocupada = limite_ocupada

    def processa_frame(self, img):
        # Converte a imagem para escala de cinza, aplica um limiar adaptativo e um filtro de mediana.
        img_cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_threshold = cv2.adaptiveThreshold(img_cinza, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)
        img_blur = cv2.medianBlur(img_threshold, 5)
        return img_cinza, img_blur

    def verifica_bancada(self, img, img_blur):
        # Verifica se a bancada está ocupada ou vazia com base na quantidade de pixels brancos na ROI.
        x, y, w, h = self.roi_bancada
        recorte = img_blur[y:y+h, x:x+w]
        qt_px_branco = cv2.countNonZero(recorte)

        if qt_px_branco > self.limite_ocupada:
            cor = (0, 0, 255)  # Vermelho
            texto = "Bancada Ocupada"
        else:
            cor = (0, 255, 0)  # Verde
            texto = "Bancada Vazia"

        cv2.rectangle(img, (x, y), (x + w, y + h), cor, 2)
        cv2.putText(img, texto, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)

        return qt_px_branco <= self.limite_ocupada