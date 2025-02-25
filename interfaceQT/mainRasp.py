from PyQt5 import uic, QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QWidget
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, QTimer, QDateTime, Qt
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np
import RPi.GPIO as IO
from algoritm import Algoritm
import time
from datetime import datetime
import threading
import sys


class SensorState(QThread):
    gpio_state_changed = pyqtSignal(bool)

    def __init__(self, button):
        super().__init__()
        self.pin = button
        self._running = False
        self.last_state = None

    def run(self):    
        self._running = True
        while self._running:
            try:
                current_state = IO.input(self.pin) == IO.LOW
                if current_state != self.last_state:
                    self.last_state = current_state
                    self.gpio_state_changed.emit(current_state)
                time.sleep(0.1)  # Atraso de 100 milissegundos
            except Exception as e:
                print(f"Erro ao ler sensor: {e}")

    def stop(self):
        self._running = False
        self.wait()

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, device):
        super().__init__()
        self._run_flag = True
        self.cap = None
        self.device = device

    def run(self):
        # self.cap = cv2.VideoCapture("/dev/video2")
        self.cap = cv2.VideoCapture(self.device)
        if not self.cap.isOpened():
            ErroCamera = "Erro ao abrir a câmera"
            return ErroCamera

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        # Máxima Resolução:
        self.WIDTH = 3840
        self.HEIGHT = 2160
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -3)

        while self._run_flag:
            ret, cv_img = self.cap.read()
            if ret:
                # print("Frame capturado:")
                # print(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                # print(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                # # time.sleep(19)
                # print(self.cap.get(cv2.CAP_PROP_FPS))
                # print(self.cap.get(cv2.CAP_PROP_BRIGHTNESS))
                # print(self.cap.get(cv2.CAP_PROP_CONTRAST))
                # print(self.cap.get(cv2.CAP_PROP_HUE))
                # print(self.cap.get(cv2.CAP_PROP_SATURATION))
                # print(self.cap.get(cv2.CAP_PROP_SHARPNESS))
                # print(self.cap.get(cv2.CAP_PROP_GAMMA))
                # print(self.cap.get(cv2.CAP_PROP_WHITE_BALANCE_RED_V))
                # print(self.cap.get(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U))
                # print(self.cap.get(cv2.CAP_PROP_EXPOSURE))
                # print(self.cap.get(cv2.CAP_PROP_GAIN))
                # print(self.cap.get(cv2.CAP_PROP_AUTOFOCUS))
                # print(self.cap.get(cv2.CAP_PROP_AUTO_WB))
                # print(self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE))
                # print("_______________________________")

                self.change_pixmap_signal.emit(cv_img)
            else:
                print("Erro ao capturar frame")
                break

        self.cap.release()

    def camParams(self):
        self.FPS = self.cap.get(cv2.CAP_PROP_FPS)
        self.WIDTH = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.HEIGHT = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        return self.FPS, self.WIDTH, self.HEIGHT

    def frame(self):
        if self.cap and self.cap.isOpened():
            result, frame = self.cap.read()
            if result:
                return frame
        return None

    def stop(self):
        self._run_flag = False
        self.wait()

    def camParams(self):
        self.FPS = self.cap.get(cv2.CAP_PROP_FPS)
        self.WIDTH = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.HEIGHT = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        return self.FPS, self.WIDTH, self.HEIGHT

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
    def __init__(self):
        super().__init__()
        self.ui = uic.loadUi("IO_page.ui", self)
        self.setWindowTitle("IO Monitor")
        self.setFixedSize(510, 200)
        
        # Set initial state of checkboxes to match the initial state of the LEDs
        # self.ui.checkBox_3.setChecked(True)
        self.ui.checkBox_4.setChecked(True)
        self.ui.checkBox_3.stateChanged.connect(lambda state, pin=23: self.handle_checkbox_state(state, pin))
        self.ui.checkBox_4.stateChanged.connect(lambda state, pin=24: self.handle_checkbox_state(state, pin))
        
        self.sensor_threads = []
        gpio_pins = [17, 27, 22]  # Lista dos pinos GPIO que voc deseja monitorar
        for pin in gpio_pins:
            thread = SensorState(pin)
            thread.gpio_state_changed.connect(self.update_GPIO_state)
            thread.start()
            self.sensor_threads.append(thread)

        self.last_gpio_states = {pin: None for pin in gpio_pins}

    def handle_checkbox_state(self, state, pin):
        if state == Qt.Checked:
            IO.output(pin, IO.HIGH)
        else:
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
        # LED sensor state visualization
        if color == "red":
            getattr(self, f"B{pin}").setStyleSheet("background-color: rgb(255, 0, 0); border-radius:15px;")
        elif color == "green":
            getattr(self, f"B{pin}").setStyleSheet("background-color: rgb(85, 255, 0); border-radius:15px;")





class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = uic.loadUi("interfaceQT.ui", self)
        self.setWindowTitle("HFA0001")

        self.opMode = None # Operation Mode (Automatic/Manual)
        self.STOP_state = None
        self.PREPARED = True
        self.auto_timer = None # Timer for AUTO mode
        self.disply_width = 3840
        self.display_height = 2160

        ##################################################
        #                   Inputs                       #
        ##################################################
        self.B01_doorLeft = IO.input(17)
        self.B02_doorRight = IO.input(27)
        self.B03_board = IO.input(22)
        self.inputs = [self.B01_doorLeft, self.B02_doorRight, self.B03_board]

        ##################################################
        #                    Pages                       #
        ##################################################
        self.ui.btn_home.clicked.connect(lambda: self.ui.stackedWidget_Pages.setCurrentWidget(self.ui.homePage))
        self.ui.btn_camara.clicked.connect(lambda: self.ui.stackedWidget_Pages.setCurrentWidget(self.ui.camaraPage))
        # IO monitor Pop-up:
        self.Win_showIO = Window_IOMonitor()

        ##################################################
        #            Buttons conections                  #
        ##################################################
        self.btn_IO.clicked.connect(self.openIOmonitor)
        self.btn_AUTO.clicked.connect(self.autoMode)
        self.btn_MANUAL.clicked.connect(self.manualMode)
        self.btn_START.clicked.connect(self.startMod)
        self.btn_STOP.clicked.connect(self.stopMod)
        self.btn_RESET.clicked.connect(self.clearTextEdit)

        ##################################################
        #                   Camara                       #
        ##################################################
        self.videoDevice = "/dev/video0"
        self.thread = VideoThread(self.videoDevice)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()
        if self.thread: # When the camera is captured, the process can begin
            self.lb_estado.setText("Pronto")
            self.btn_START.setEnabled(True)
        
        ##################################################
        #                  Relogio                       #
        ##################################################
        self.lcd_timer = QTimer()
        self.lcd_timer.timeout.connect(self.clock)
        self.lcd_timer.start() 

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img): # refresh frame and represent in label
        qt_img = self.convert_cv_qt(cv_img)
        self.lb_videoImage.setPixmap(qt_img)
        # Display Params of camera
        self.FPS, self.WIDTH, self.HEIGHT = self.thread.camParams()
        self.lb_FPS.setText(f"{self.FPS}")
        self.lb_WIDTH.setText(f"{self.WIDTH}")
        self.lb_HEIGHT.setText(f"{self.HEIGHT}")


    def convert_cv_qt(self, cv_img): # Convert OpenCV format to QT format for visualization
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
        self.STOP_state = False
        warnings = {
            self.B01_doorLeft: "WARN002 - Porta Esquerda Aberta",
            self.B02_doorRight: "WARN003 - Porta Direita Aberta",
            self.B03_board: "WARN004 - Sem presença de Tabuleiro"
        }
        all_pressed = True
        warning_messages = []
        print(self.inputs)
        for self.inputs, warning in warnings.items():
            if self.inputs:
                all_pressed = False
                time_str = self.DateTime.toString('hh:mm:ss')
                warning_messages.append(f"{time_str}   {warning}")

        if warning_messages:
            self.textEdit.setTextColor(QtGui.QColor("orange"))
            for message in warning_messages:
                self.textEdit.append(message)
            self.textEdit.setTextColor(QtGui.QColor("black"))

        if self.opMode == "MANUAL":
            if all_pressed:
                while not self.STOP_state:
                    self.btn_START.setEnabled(False)
                    self.btn_STOP.setEnabled(True)
                    self.lb_estado.setText("A Processar")
                    self.analiseProcess()
                    self.STOP_state = True

        elif self.opMode == "AUTO":
            if all_pressed:
                self.auto_mode_operation()
            elif self.B03_board:
                self.start_auto_timer()

        else:
            time_str = self.DateTime.toString('hh:mm:ss')
            self.textEdit.setTextColor(QtGui.QColor("orange"))
            self.textEdit.append(f"{time_str}   WARN001 - Nenhum modo de operação selecionado")
            self.textEdit.setTextColor(QtGui.QColor("black"))
        
        self.lb_estado.setText("Pronto")
        self.btn_START.setEnabled(True)
        self.btn_STOP.setEnabled(False)

    def auto_mode_operation(self):
        self.btn_START.setEnabled(False)
        self.btn_STOP.setEnabled(True)
        self.lb_estado.setText("A Processar")
        self.analiseProcess()
        self.PREPARED = True

    def start_auto_timer(self):
        if self.auto_timer:
            self.auto_timer.cancel()

        self.PREPARED = False
        self.auto_timer = threading.Timer(5.0, self.set_prepared_true)
        self.auto_timer.start()

    def set_prepared_true(self):
        self.PREPARED = True
        self.auto_mode_operation()

    def stopMod(self):
        self.btn_START.setEnabled(True)
        self.btn_STOP.setEnabled(False)
        self.STOP_state = True
        if self.opMode == "MANUAL":
            self.btn_MANUAL.setEnabled(False)
            self.btn_AUTO.setEnabled(True)
        else:
            self.btn_MANUAL.setEnabled(True)
            self.btn_AUTO.setEnabled(False)

    def clearTextEdit(self):
        self.textEdit.clear()

    def analiseProcess(self):
        algoritm = Algoritm()

        self.ti = time.time() # Start time
        IO.output(23, IO.HIGH)
        time.sleep(3)
        # White Ilumination Image Aquisition
        whiteImage = self.thread.frame()
        current_time = datetime.fromtimestamp(self.ti)
        minutes = current_time.strftime("%M")
        cv2.imwrite(f"Image1_{minutes}.png", whiteImage)
        IO.output(23, IO.LOW) # Turn of white ilumination
        time.sleep(2) # delay for camera stabilize

        # White Ilumination Image Aquisition
        UVImage = self.thread.frame()
        cv2.imwrite(f"Image2_{minutes}.png", UVImage)

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

    ##################################################
    #            Outputs & Inputs                    #
    ##################################################
    # IO Board
    IO_INPUT = [17,27,22]
    IO_OUTPUT = [23,24]
    IO.setwarnings(False)
    IO.setmode(IO.BCM)
    
    for input_pins in IO_INPUT:
        IO.setup(input_pins, IO.IN)
        
    for output_pins in IO_OUTPUT:
        IO.setup(output_pins, IO.OUT, initial=IO.LOW)
    
    IO.output(24, IO.HIGH)

    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())

