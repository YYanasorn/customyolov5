from socket import *

HOST = '192.168.3.10'
PORT = 502


server = socket(AF_INET, SOCK_STREAM)
server.connect((HOST, PORT))

server.send(b'Hello')
print(server.recv(1024))
