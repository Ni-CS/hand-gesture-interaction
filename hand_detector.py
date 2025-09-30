import cv2
from ultralytics import YOLO

class HandDetector:
    """
    Classe responsável por detectar key-points da mão em uma imagem
    usando o modelo YOLO treinado.
    """
    def __init__(self, model_path):
        """
        Inicializa o detector, carregando o modelo YOLO.
        :param model_path: Caminho para o arquivo do modelo (.pt).
        """
        self.model = YOLO(model_path)
        print("Modelo YOLO de detecção de mãos carregado com sucesso.")

    def detect(self, frame):
        """
        Detecta a mão em um quadro de vídeo e retorna os pontos-chave.
        :param frame: O quadro da imagem (formato OpenCV/numpy).
        :return: Uma tupla contendo (keypoints, annotated_frame).
                 - keypoints: Coordenadas dos pontos-chave da primeira mão detectada, ou None.
                 - annotated_frame: O quadro original com as detecções desenhadas.
        """
        # Executa a inferência do modelo no quadro
        results = self.model(frame, verbose=False)

        # Desenha as anotações no quadro para visualização
        annotated_frame = results[0].plot()

        # Extrai os pontos-chave
        keypoints = None
        if results[0].keypoints and results[0].keypoints.shape[1] > 0:
            # Pega as coordenadas (x, y) dos pontos da primeira mão detectada
            keypoints = results[0].keypoints.xy[0]

        return keypoints, annotated_frame