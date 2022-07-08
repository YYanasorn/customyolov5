import serial
import time

ser = serial.Serial("COM3", 9600)
while True:
    input_value = "0"
    print(input_value)
    ser.write(input_value.encode())
    time.sleep(2)

    input_value = "1"
    print(input_value)
    ser.write(input_value.encode())
    time.sleep(2)