from hashlib import new
from PyQt5 import uic, QtCore, QtGui, QtWidgets
# from PyQt5.QtMultimedia import QCameraInfo
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QWidget
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, QTimer, QDateTime, Qt
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
import cv2,time,sys
import numpy as np
import random as rnd
import RPi.GPIO as IO
import time
import numpy as np
from algoritm import Algoritm
from imageAcq import ImageAcq

class SensorState(QThread):
    gpio_state_changed = pyqtSignal(bool)
    
    def __init__(self, pin):
        super().__init__()
        self._running = False
        self.pin = pin
        self.last_state = None
     
    def run(self):    
        self._running = True
        while self._running:
            try:
                # Verificar o estado atual do pino GPIO
                current_state = IO.input(self.pin) == IO.LOW
                
                # Verificar se houve uma mudana de estado
                if current_state != self.last_state:
                    # Atualizar o estado anterior
                    self.last_state = current_state
                    # Emitir o sinal apenas se houve uma mudana de estado
                    self.gpio_state_changed.emit(current_state)
                
                # Aguardar um curto perodo antes da prxima leitura
                time.sleep(0.1)  # Atraso de 100 milissegundos
            except Exception as e:
                print(f"Erro ao ler sensor: {e}")
                # Se ocorrer um erro, pare a thread para evitar loop infinito
    
    def stop(self):
        """Parar a thread"""
        self._running = False
        self.wait()  # Aguardar a thread terminar antes de retornar


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    
    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.cap = None

    def run(self):    
        # capture from web cam
        self.cap = cv2.VideoCapture("/dev/video0")

        # Verificar se a câmera foi aberta com sucesso
        if not self.cap.isOpened():
            print("Erro ao abrir a câmera")
            return

        self.cap.set(cv2.CAP_PROP_FPS, 10)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, 156)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840.0)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160.0)
        
        print("Configurações da câmera:")
        print(self.cap.get(cv2.CAP_PROP_FPS))
        print(self.cap.get(cv2.CAP_PROP_BRIGHTNESS))
        print(self.cap.get(cv2.CAP_PROP_CONTRAST))
        print(self.cap.get(cv2.CAP_PROP_HUE))
        print(self.cap.get(cv2.CAP_PROP_SATURATION))
        print(self.cap.get(cv2.CAP_PROP_SHARPNESS))
        print(self.cap.get(cv2.CAP_PROP_GAMMA))
        print(self.cap.get(cv2.CAP_PROP_WHITE_BALANCE_RED_V))
        print(self.cap.get(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U))
        print(self.cap.get(cv2.CAP_PROP_EXPOSURE))
        print(self.cap.get(cv2.CAP_PROP_GAIN))

        while self._run_flag:  # Loop enquanto a flag de execução estiver True
            ret, cv_img = self.cap.read()
            if ret:
                print("Frame capturado:")
                print(self.cap.get(cv2.CAP_PROP_BRIGHTNESS))
                print(self.cap.get(cv2.CAP_PROP_CONTRAST))
                print(self.cap.get(cv2.CAP_PROP_HUE))
                print(self.cap.get(cv2.CAP_PROP_SATURATION))
                print(self.cap.get(cv2.CAP_PROP_SHARPNESS))
                print(self.cap.get(cv2.CAP_PROP_GAMMA))
                print(self.cap.get(cv2.CAP_PROP_WHITE_BALANCE_RED_V))
                print(self.cap.get(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U))
                print(self.cap.get(cv2.CAP_PROP_EXPOSURE))
                print(self.cap.get(cv2.CAP_PROP_GAIN))
                print("_______________________________")

                self.change_pixmap_signal.emit(cv_img)
            else:
                print("Erro ao capturar frame")
                break
        
        self.cap.release()
    
    def frame(self):
        if self.cap and self.cap.isOpened():
            # Capturar apenas um frame
            result, frame = self.cap.read()
            if result:
                return frame
        return None
    
    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()

class Window_IOMonitor(QDialog):
    
    def __init__(self):
        super().__init__()
        self.ui = uic.loadUi("IO_page.ui", self)
        self.setWindowTitle("IO Monitor")
        self.setFixedSize(510, 200)

        # Conectar o sinal stateChanged da checkBox_3 ao método handle_checkbox_state para o pino 16
        self.ui.checkBox_3.stateChanged.connect(lambda state, pin=16: self.handle_checkbox_state(state, pin))
        # Conectar o sinal stateChanged da checkBox_4 ao método handle_checkbox_state para o pino 18
        self.ui.checkBox_4.stateChanged.connect(lambda state, pin=18: self.handle_checkbox_state(state, pin))
        
        # Lista para armazenar as threads de monitoramento de GPIO
        self.sensor_threads = []
        
        # Instanciando as threads para monitorar os pinos GPIO desejados
        gpio_pins = [11, 13, 15]  # Lista dos pinos GPIO que voc deseja monitorar
        for pin in gpio_pins:
            thread = SensorState(pin)
            thread.gpio_state_changed.connect(self.update_GPIO_state)
            thread.start()
            self.sensor_threads.append(thread)
        
        # Dicionrio para armazenar o ltimo estado de cada pino GPIO
        self.last_gpio_states = {pin: None for pin in gpio_pins}

    def handle_checkbox_state(self, state, pin):
        # Verificar se a caixa de seleção está marcada
        if state == Qt.Checked:
            # Se estiver marcada, definir o pino de saída como HIGH (ativo)
            IO.output(pin, IO.HIGH)
        else:
            # Se não estiver marcada, definir o pino de saída como LOW (inativo)
            IO.output(pin, IO.LOW)
               
    @pyqtSlot(bool)
    def update_GPIO_state(self, gpio_state):
        # Identificar de qual thread o sinal foi emitido
        sender_thread = self.sender()
        # Encontrar o ndice da thread na lista de threads
        thread_index = self.sensor_threads.index(sender_thread)
        # Identificar o pino GPIO correspondente ao ndice
        pin = sender_thread.pin
        
        # Verificar se houve uma mudana de estado
        if gpio_state != self.last_gpio_states[pin]:
            # Atualizar o estado anterior
            self.last_gpio_states[pin] = gpio_state
            
            # Atualizar a interface do usurio de acordo com o estado do pino GPIO
            if gpio_state:
                self.set_gpio_indicator_color(pin, "red")
            else:
                self.set_gpio_indicator_color(pin, "green")
    
    def set_gpio_indicator_color(self, pin, color):
        # Mtodo para definir a cor do indicador de GPIO na interface do usurio
        if color == "red":
            getattr(self, f"B{pin}").setStyleSheet("background-color: rgb(255, 0, 0); border-radius:15px;")
        elif color == "green":
            getattr(self, f"B{pin}").setStyleSheet("background-color: rgb(85, 255, 0); border-radius:15px;")

class MainWindow(QMainWindow):
        
    def __init__(self):
        super().__init__()
        self.ui = uic.loadUi("interfaceQT.ui",self)
        self.setWindowTitle("HFA0001")

        # Variables
        self.opMode = None # Operation mode variable
        
        # Pages
        self.ui.btn_home.clicked.connect(lambda: self.ui.stackedWidget_Pages.setCurrentWidget(self.ui.homePage))
        self.ui.btn_camara.clicked.connect(lambda: self.ui.stackedWidget_Pages.setCurrentWidget(self.ui.camaraPage))
        # self.ui.btn_home.clicked.connect(self.stopVideo)
        # self.ui.btn_camara.clicked.connect(self.startVideo)

        
        # Create Instance class
        self.Win_showIO = Window_IOMonitor()

        # Clock
        self.lcd_timer = QTimer()
        self.lcd_timer.timeout.connect(self.clock)
        self.lcd_timer.start() 
        
        # Home page
        #Buttons
        self.btn_STOP.setEnabled(False)

        self.btn_IO.clicked.connect(self.openIOmonitor)
        self.btn_AUTO.clicked.connect(self.autoMode)
        self.btn_MANUAL.clicked.connect(self.manualMode)
        self.btn_START.clicked.connect(self.startMod)
        self.btn_STOP.clicked.connect(self.stopMod)
        self.btn_RESET.clicked.connect(self.clearTextEdit)

        # Create Instance class for video thread
        self.disply_width = 3840
        self.display_height = 2160
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()



    # def startVideo(self):
    #     self.thread.change_pixmap_signal.connect(self.update_image)
    #     self.thread.start()

    # def stopVideo(self):
    #     try:
    #         # Desconecta o sinal de mudança de pixmap
    #         self.thread.change_pixmap_signal.disconnect()
    #         # Sinaliza para o thread parar
    #         self.thread.stop()
    #         # Aguarda o término do thread
    #         self.thread.wait()
    #     except:
    #         pass  # Ignora o erro se o sinal não estiver conectado

            
    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.lb_videoImage.setPixmap(qt_img)
    
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
        
        
    def clock(self):
        self.DateTime = QDateTime.currentDateTime()
        self.lcd_clock.display(self.DateTime.toString('hh:mm'))

    # Open I/O Window monitor 
    def openIOmonitor(self):
        self.Win_showIO.show()

    def manualMode(self):
        self.opMode = "MANUAL"
        self.lb_opMode.setText(self.opMode)
        self.btn_MANUAL.setEnabled(False)
        self.btn_AUTO.setEnabled(True)

    def autoMode(self):
        self.opMode = "AUTO"
        self.lb_opMode.setText(self.opMode)
        self.btn_AUTO.setEnabled(False)
        self.btn_MANUAL.setEnabled(True)

    def startMod(self):
        if self.opMode == "MANUAL":
            cameraShoot = ImageAcq()
            algoritm = Algoritm()

            # UV Image capture
            UVImage = self.thread.frame()
            cv2.imwrite("Image2.png", UVImage)

            # White Image capture
            IO.output(16, IO.HIGH)
            time.sleep(10)
            whiteImage = self.thread.frame()
            IO.output(16, IO.LOW)
            cv2.imwrite("Image1.png", whiteImage)
            
            try:
                self.ti = time.time()
                final_result = algoritm.main(whiteImage, UVImage)
                # original = cv2.imread('../interfaceQT/static/image.png')
                if final_result is not None:
                   # Convert the OpenCV image (numpy.ndarray) to a QImage
                   height, width, channel = final_result.shape
                   bytesPerLine = 3 * width
                   qImg = QImage(final_result.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()      
                   # Convert the QImage to a QPixmap and set it
                   pixmap = QPixmap.fromImage(qImg)
                   self.lb_resultImage.setPixmap(pixmap)
                   self.lb_resultImage.setScaledContents(True)
                   self.tf = time.time()
                   self.lb_cicleTime.setText(str(round(self.tf-self.ti, 2)))
  
            except:
                time_str = self.DateTime.toString('hh:mm:ss')
                self.textEdit.setTextColor(QtGui.QColor("red")) 
                self.textEdit.append(f"{time_str}   ALARM002 - Problemas a carregar a imagem final")
                self.textEdit.setTextColor(QtGui.QColor("black"))
                pass
            # height, width, channel = whiteImage.shape
            # bytesPerLine = 3 * width
            # qImg = QImage(whiteImage.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()      
            # # Convert the QImage to a QPixmap and set it
            # pixmap = QPixmap.fromImage(qImg)
            # self.lb_resultImage.setPixmap(pixmap)
            # self.lb_resultImage.setScaledContents(True)
            # # self.lb_cicleTime.setText(str(round(self.tf-self.ti, 2)))

        elif self.opMode == "AUTO":
            self.textEdit.append("EM MODO AUTO")

        else:
            time_str = self.DateTime.toString('hh:mm:ss')
            self.textEdit.setTextColor(QtGui.QColor("orange")) 
            self.textEdit.append(f"{time_str}   WARN001 - Nenhum modo de operação selecionado")
            self.textEdit.setTextColor(QtGui.QColor("black"))


        # if self.opMode == None:
        #     time_str = self.DateTime.toString('hh:mm:ss')
        #     self.textEdit.setTextColor(QtGui.QColor("orange")) 
        #     self.textEdit.append(f"{time_str}   WARN001 - Nenhum modo de operação selecionado")
        #     self.textEdit.setTextColor(QtGui.QColor("black"))
        # else:
        #     time_str = self.DateTime.toString('hh:mm:ss')
        #     self.textEdit.append(f"{time_str} - Modo Start ativo")
        #     self.btn_START.setEnabled(False)
        #     self.btn_STOP.setEnabled(True)
        #     self.btn_AUTO.setEnabled(False)
        #     self.btn_MANUAL.setEnabled(False)
        #     self.ti = time.time()
        #     algoritm = Algoritm()
        #     imageWhite = cv2.imread('static/Image1.png')
        #     imageUV = cv2.imread('static/Image2.png')
        #     final_result = algoritm.main(imageWhite, imageUV)
        #     # original = cv2.imread('../interfaceQT/static/image.png')
        #     if final_result is not None:
        #        # Convert the OpenCV image (numpy.ndarray) to a QImage
        #        height, width, channel = final_result.shape
        #        bytesPerLine = 3 * width
        #        qImg = QImage(final_result.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()      
        #        # Convert the QImage to a QPixmap and set it
        #        pixmap = QPixmap.fromImage(qImg)
        #        self.lb_resultImage.setPixmap(pixmap)
        #        self.lb_resultImage.setScaledContents(True)
        #        self.tf = time.time()
        #        self.lb_cicleTime.setText(str(round(self.tf-self.ti, 2)))
        #     else:
        #        time_str = self.DateTime.toString('hh:mm:ss')
        #        self.textEdit.setTextColor(QtGui.QColor("red")) 
        #        self.textEdit.append(f"{time_str}   ALARM002 - Problemas a carregar a imagem final")
        #        self.textEdit.setTextColor(QtGui.QColor("black"))

        
    def stopMod(self):
        if self.opMode == "MANUAL":
            self.btn_MANUAL.setEnabled(False)
            self.btn_AUTO.setEnabled(True)
        else:
            self.btn_MANUAL.setEnabled(True)
            self.btn_AUTO.setEnabled(False)
        self.btn_START.setEnabled(True)
        self.btn_STOP.setEnabled(False)

    def clearTextEdit(self):
        self.textEdit.clear()  


if __name__ == "__main__":
    
    # IO Board
    IO_INPUT = [11,13,15]
    IO_OUTPUT = [16,18]
    IO.setwarnings(False)
    IO.setmode(IO.BOARD)
    
    for input_pins in IO_INPUT:
        IO.setup(input_pins, IO.IN)
        
    for output_pins in IO_OUTPUT:
        IO.setup(output_pins, IO.OUT, initial=IO.LOW)
    
    IO.output(18, IO.HIGH)

        
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
