from socket import *

HOST = "192.168.3.10"
PORT = 502

server = socket(AF_INET, SOCK_STREAM)
server.bind((HOST, PORT))
server.listen(5)


while True:
    communication_socket, address = server.accept()
    print(f"Connected to {address}")
    message = communication_socket.recv(1024).decode()
    print(f"Message from client is : {message}")
    communication_socket.send(f"Got your message! thank you!".encode())
    communication_socket.close()
    print(f"Connection with {address} ended!")
