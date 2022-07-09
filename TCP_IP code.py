from pymodbus.client.sync import ModbusTcpClient as ModbusClient

PLC_IP = "" # PLC IP Address 
PLC_Port = 502 #TCP PORT
client = ModbusClient(PLC_IP, PLC_Port)
client.connect()
UNIT = 0x1

print("Connect")