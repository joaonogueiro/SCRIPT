#!/usr/bin/env python3

from pymodbus.client import ModbusTcpClient

# Configura o cliente MODBUS com o endereço IP do dispositivo
client = ModbusTcpClient('192.168.1.20')

# Conecta ao servidor MODBUS
client.connect()

# Endereço do sinal digital (coil) que você quer escrever
coil_address = 16

# Valor que você quer escrever (True para ligar, False para desligar)
value = False

# Escreve no coil
write_response = client.write_coil(coil_address, value)

# Verifica se a escrita foi bem-sucedida
if write_response.isError():
    print(f"Erro ao escrever no endereço {coil_address}")
else:
    print(f"Escrita bem-sucedida no endereço {coil_address}")

# Fecha a conexão
client.close()
