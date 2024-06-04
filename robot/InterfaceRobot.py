from PyQt5 import uic, QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QWidget
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, QTimer, QDateTime, Qt
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
import time

from pymodbus.client import ModbusTcpClient


class MainWindow(QMainWindow):
        
    def __init__(self):
        super().__init__()
        self.ui = uic.loadUi("InterfaceRobot.ui",self)
        self.setWindowTitle("HFA0001_ROBOT")
        self.mod = 0

        self.ui.btn_START.clicked.connect(self.operationMod)

        # Fecha a conexão
        client.close()


    def operationMod(self):
        if self.btn_START.text() == "START":
            value = 1
            self.btn_START.setText('STOP')
        else:
            value = 0
            self.btn_START.setText('START')

        # Endereço do sinal digital (coil) que você quer escrever
        coil_address = 16

        # Escreve no coil
        write_response = client.write_coil(coil_address, value)
        # response = client.write_register(2, 89)

        # Verifica se a escrita foi bem-sucedida
        if write_response.isError():
            print(f"Erro ao escrever no endereço {coil_address}")
        else:
            print(f"Escrita bem-sucedida no endereço {coil_address}")


if __name__ == "__main__":
    # Configura o cliente MODBUS com o endereço IP do dispositivo
    client = ModbusTcpClient('192.168.1.20')

    # Conecta ao servidor MODBUS
    client.connect()
           
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
