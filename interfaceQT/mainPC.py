from PyQt5 import uic, QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QWidget
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, QTimer, QDateTime, Qt
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np
from gpiozero import LED, Button
from gpiozero.pins.pigpio import PiGPIOFactory
from algoritm import Algoritm
from imageAcq import ImageAcq
import time

class SensorState(QThread):
    gpio_state_changed = pyqtSignal(bool)
    
    def __init__(self, pin, pi):
        super().__init__()
        # print(f"Inicializando SensorState com pino {pin}")
        self.pin = Button(pin, pin_factory=pi)
        # print("Pin inicializado:", self.pin)
        self._running = False
        self.last_state = None
     
    def run(self):    
        self._running = True
        while self._running:
            try:
                current_state = self.pin.is_pressed
                if current_state != self.last_state:
                    self.last_state = current_state
                    self.gpio_state_changed.emit(current_state)
                time.sleep(0.1)
            except Exception as e:
                print(f"Erro ao ler sensor: {e}")

    def stop(self):
        self._running = False
        self.wait()

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    
    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.cap = None

    def run(self):
        self.cap = cv2.VideoCapture("/dev/video0")
        if not self.cap.isOpened():
            print("Erro ao abrir a câmera")
            return
        
        # Máxima Resolução:
        self.WIDTH = 3840
        self.HEIGHT = 2160
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, 10)


        # self.cap.set(cv2.CAP_PROP_EXPOSURE, 156)
        # self.cap.set(cv2.CAP_PROP_SATURATION, 64)
        # self.cap.set(cv2.CAP_PROP_GAIN, 64)

        # # # Default values:
        # self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 0)
        # self.cap.set(cv2.CAP_PROP_CONTRAST, 0)
        # self.cap.set(cv2.CAP_PROP_SATURATION, 64)
        # self.cap.set(cv2.CAP_PROP_GAIN, 100)
        # self.cap.set(cv2.CAP_PROP_EXPOSURE, 156)
        # self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        # self.cap.set(cv2.CAP_PROP_AUTO_WB, 1)
        # self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)




        # Params de teste
        # self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 0)
        # self.cap.set(cv2.CAP_PROP_CONTRAST, 0)
        # self.cap.set(cv2.CAP_PROP_SATURATION, 40)
        # self.cap.set(cv2.CAP_PROP_GAIN, 90)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -13)
        # self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        # self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)





        while self._run_flag:
            ret, cv_img = self.cap.read()
            if ret:
                print("Frame capturado:")
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
                print(self.cap.get(cv2.CAP_PROP_AUTOFOCUS))
                print(self.cap.get(cv2.CAP_PROP_AUTO_WB))
                print(self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE))
                print("_______________________________")

                self.change_pixmap_signal.emit(cv_img)
            else:
                print("Erro ao capturar frame")
                break
        
        self.cap.release()
    
    def frame(self):
        if self.cap and self.cap.isOpened():
            result, frame = self.cap.read()
            if result:
                return frame
        return None
    
    def stop(self):
        self._run_flag = False
        self.wait()

class Window_IOMonitor(QDialog):
    
    def __init__(self, pi, led_16, led_18):
        super().__init__()
        self.ui = uic.loadUi("IO_page.ui", self)
        self.setWindowTitle("IO Monitor")
        self.setFixedSize(510, 200)
        self.pi = pi

        self.led_16 = led_16
        self.led_18 = led_18
        # self.led_16 = LED(16, pin_factory=self.pi)
        # self.led_18 = LED(18, pin_factory=self.pi)

        self.ui.checkBox_3.stateChanged.connect(lambda state: self.handle_checkbox_state(state, self.led_16))
        self.ui.checkBox_4.stateChanged.connect(lambda state: self.handle_checkbox_state(state, self.led_18))
        
        self.sensor_threads = []
        gpio_pins = [17, 27, 22]
        for pin in gpio_pins:
            thread = SensorState(pin, self.pi)
            thread.gpio_state_changed.connect(self.update_GPIO_state)
            thread.start()
            self.sensor_threads.append(thread)
        
        self.last_gpio_states = {pin: None for pin in gpio_pins}

    def handle_checkbox_state(self, state, led):
        if state == Qt.Checked:
            led.on()
        else:
            led.off()
               
    @pyqtSlot(bool)
    def update_GPIO_state(self, gpio_state):
        sender_thread = self.sender()
        thread_index = self.sensor_threads.index(sender_thread)
        pin = sender_thread.pin.pin.number
        
        if gpio_state != self.last_gpio_states[pin]:
            self.last_gpio_states[pin] = gpio_state
            
            if gpio_state:
                self.set_gpio_indicator_color(pin, "red")
            else:
                self.set_gpio_indicator_color(pin, "green")
    
    def set_gpio_indicator_color(self, pin, color):
        if color == "red":
            getattr(self, f"B{pin}").setStyleSheet("background-color: rgb(255, 0, 0); border-radius:15px;")
        elif color == "green":
            getattr(self, f"B{pin}").setStyleSheet("background-color: rgb(85, 255, 0); border-radius:15px;")

class MainWindow(QMainWindow):
        
    def __init__(self, pi):
        super().__init__()
        self.ui = uic.loadUi("interfaceQT.ui", self)
        self.setWindowTitle("HFA0001")
        self.pi = pi

        # Iluminação
        led_16 = LED(23, pin_factory=self.pi)
        led_18 = LED(24, pin_factory=self.pi)
        self.led_16 = led_16
        self.led_18 = led_18
        self.led_18.on()
        self.led_16.on()

        # Sensores
        self.pin17 = Button(17, pin_factory=pi)
        self.pin27 = Button(27, pin_factory=pi)
        self.pin22 = Button(27, pin_factory=pi)


        self.opMode = None
        
        self.ui.btn_home.clicked.connect(lambda: self.ui.stackedWidget_Pages.setCurrentWidget(self.ui.homePage))
        self.ui.btn_camara.clicked.connect(lambda: self.ui.stackedWidget_Pages.setCurrentWidget(self.ui.camaraPage))

        self.Win_showIO = Window_IOMonitor(self.pi, self.led_16, self.led_18)

        self.lcd_timer = QTimer()
        self.lcd_timer.timeout.connect(self.clock)
        self.lcd_timer.start() 
        
        self.btn_STOP.setEnabled(False)

        self.btn_IO.clicked.connect(self.openIOmonitor)
        self.btn_AUTO.clicked.connect(self.autoMode)
        self.btn_MANUAL.clicked.connect(self.manualMode)
        self.btn_START.clicked.connect(self.startMod)
        self.btn_STOP.clicked.connect(self.stopMod)
        self.btn_RESET.clicked.connect(self.clearTextEdit)

        self.disply_width = 3840
        self.display_height = 2160
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.lb_videoImage.setPixmap(qt_img)
    
    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
        
    def clock(self):
        self.DateTime = QDateTime.currentDateTime()
        self.lcd_clock.display(self.DateTime.toString('hh:mm'))

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
            while True:
                if not self.pin17.is_pressed and not self.pin27.is_pressed and not self.pin22.is_pressed:
                    print("Tudo em baixo")
            # cameraShoot = ImageAcq()
            # algoritm = Algoritm()

            # self.ti = time.time()
            # whiteImage = self.thread.frame()
            # self.led_16.off()
            # cv2.imwrite("Image1.png", whiteImage)
            # time.sleep(16)
            # UVImage = self.thread.frame()
            # cv2.imwrite("Image2.png", UVImage)
            # self.led_16.on()

            # Debug Image visualitazion
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

    def autoOperation(self):
        print("ola")
        
    def algoritmImpl(self, whiteImage, UVImage):
        algoritm = Algoritm()
        try:
            final_result = algoritm.main(whiteImage, UVImage)
            cv2.imwrite("FINALImg.png", final_result)
            # if final_result is not None:
            height, width, channel = final_result.shape
            bytesPerLine = 3 * width
            qImg = QImage(final_result.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(qImg)
            self.lb_resultImage.setPixmap(pixmap)
            self.lb_resultImage.setScaledContents(True)
            self.tf = time.time()
            self.lb_cicleTime.setText(str(round(self.tf - self.ti, 2)))
  
        except Exception as e:
                print(f"Erro ao processar imagem: {e}")
                time_str = self.DateTime.toString('hh:mm:ss')
                self.textEdit.setTextColor(QtGui.QColor("red")) 
                self.textEdit.append(f"{time_str}   ALARM002 - Problemas a carregar a imagem final")
                self.textEdit.setTextColor(QtGui.QColor("black"))
                pass
 

if __name__ == "__main__":
   
   pi_factory = PiGPIOFactory(host='10.0.2.128')
   app = QApplication([])
   window = MainWindow(pi_factory)
   window.show()
   app.exec_()

