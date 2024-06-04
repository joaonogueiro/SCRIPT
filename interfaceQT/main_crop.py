import cv2
import numpy as np

# Variáveis globais para armazenar pontos de clique e a escala da imagem
points = []
scale = 3

# Função de callback para eventos de mouse
def click_event(event, x, y, flags, param):
    global points
    # Se o botão esquerdo do mouse for clicado
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(small_img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Image", small_img)
        
        # Se dois pontos forem clicados, recortar a imagem
        if len(points) == 2:
            crop_image()

def crop_image():
    global points
    # Ajustar pontos para a escala da imagem original
    x1, y1 = int(points[0][0] * scale), int(points[0][1] * scale)
    x2, y2 = int(points[1][0] * scale), int(points[1][1] * scale)
    
    # Definir a região de interesse (ROI)
    roi = img[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]
    
    # Mostrar a imagem recortada
    cv2.imshow("Cropped Image", roi)
    cv2.imwrite("../interfaceQT/testImages/cropped_image.png", roi)
    points = []  # Resetar pontos

# Carregar a imagem
img = cv2.imread("../interfaceQT/testImages/imagesAcq/Image1_31.png")
h, w = img.shape[:2]

# Redimensionar a imagem para ser 3 vezes menor
small_img = cv2.resize(img, (w // scale, h // scale))

cv2.imshow("Image", small_img)

# Configurar a função de callback para eventos de mouse
cv2.setMouseCallback("Image", click_event)

# Manter a janela aberta até a tecla ESC ser pressionada
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Tecla ESC
        break

cv2.destroyAllWindows()
