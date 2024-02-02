from socket import *

serverPort = 30000
serverSocket = socket(AF_INET, SOCK_DGRAM)

serverSocket.bind(("", serverPort))
print("The Server is ready to receive requests")

while True:
    message, clientAddress = serverSocket.recvfrom(2048)
    modifiedMessage = message.decode().upper()
    serverSocket.sendto(modifiedMessage.encode(),clientAddress)
