import cv2
import time
class ImageAcq:
    def __init__(self):
        self.devicePath = '/dev/video0' 

        # Máxima Resolução:
        self.WIDTH = 3840
        self.HEIGHT = 2160
    
    def cameraShooting(self, mod):
        # Inicialização
        cap = cv2.VideoCapture(self.devicePath) 
        if mod == 1:
            # Ajusta a 
            time.sleep(5)
            cap.set(cv2.CAP_PROP_EXPOSURE, 0)  # Ajuste esse valor conforme necessário

            # Ajusta o ganho
            # cap.set(cv2.CAP_PROP_GAIN, 10)  # Ajuste esse valor conforme necessário

            # Ajusta o balanço de brancos automático
            # cap.set(cv2.CAP_PROP_AUTO_WB, True)
        # cap.set(cv2.CAP_PROP_EXPOSURE, 0)  # Ajuste esse valor conforme necessário
        elif mod == 0:
            # Ajusta a exposição
            # cap.set(cv2.CAP_PROP_BRIGHTNESS, 0)  
            # cap.set(cv2.CAP_PROP_CONTRAST, 0)    # Contraste padrão
            # cap.set(cv2.CAP_PROP_SATURATION, 64)  # Saturação padrão
            # cap.set(cv2.CAP_PROP_HUE, 0)           # Matiz padrão
            # cap.set(cv2.CAP_PROP_GAIN, 100)          # Ganho padrão
            # cap.set(cv2.CAP_PROP_EXPOSURE, 156)     # Exposição automática # Ajuste esse valor conforme necessário
            # cap.set(cv2.CAP_PROP_EXPOSURE, 156)     # Exposição automática # Ajuste esse valor conforme necessário
            
            
            # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840.0)
            # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160.0)
            cap.set(cv2.CAP_PROP_EXPOSURE, 156)
            cap.set(cv2.CAP_PROP_FPS, 5.0)
            time.sleep(5)
            # cap.set(cv2.CAP_PROP_POS_MSEC, 0.0)
            # cap.set(cv2.CAP_PROP_FRAME_COUNT, -1.0)
            # cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.0)
            # cap.set(cv2.CAP_PROP_CONTRAST, 0.0)
            # cap.set(cv2.CAP_PROP_SATURATION, 64.0)
            # cap.set(cv2.CAP_PROP_HUE, 0.0)
            # cap.set(cv2.CAP_PROP_GAIN, 100.0)
            # cap.set(cv2.CAP_PROP_CONVERT_RGB, 1.0)
            # cap.set(cv2.CAP_PROP_EXPOSURE, 156.0)

            print(cap.get(cv2.CAP_PROP_BRIGHTNESS),  
            cap.get(cv2.CAP_PROP_BRIGHTNESS),  
            cap.get(cv2.CAP_PROP_CONTRAST),    
            cap.get(cv2.CAP_PROP_SATURATION),  
            cap.get(cv2.CAP_PROP_HUE),           
            cap.get(cv2.CAP_PROP_GAIN))

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.HEIGHT)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
        # Capturar apenas um frame
        result, frame = cap.read() 

        # Liberar o objeto de captura
        cap.release()

        # Verificar se a captura foi bem-sucedida
        if result:
            return frame
        else:
            return None

# # Exemplo de uso:
# image_acq = ImageAcq()
# frame = image_acq.cameraShooting()

# if frame is not None:
#     # Resize the image 
#     scale_factor = 0.25
#     cap_resize = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)
#     # Aqui você pode fazer o que quiser com o frame, por exemplo, exibi-lo
#     cv2.imshow('Frame', cap_resize)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# else:
#     print("Erro ao capturar o frame.")