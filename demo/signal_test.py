import signal
import sys
import socket


# signal.pause()

class Server:

    def __init__(self, server_address, buffer_size=1024 * 1024):

        self.buffer_size = buffer_size
        self.socket_TCP = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket_TCP.bind(server_address)
        self.socket_TCP.listen(1)

    def run(self):
        print("TCP server is running...")
        while True:
            connection, client_address = self.socket_TCP.accept()

    def signal_handler(self, sig, frame):
        print('You pressed Ctrl+C!')
        sys.exit(0)


s = Server(('127.0.0.1', 5050))
s.run()
