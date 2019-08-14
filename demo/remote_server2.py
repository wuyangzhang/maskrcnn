import socket
import sys
sys.path.append('/home/nvidia/maskrcnn-benchmark')
import pickle
import pickle
import struct
import datetime, time

from app.app_manager import ApplicationManager


class Server:

    def __init__(self, server_address, buffer_size=1024 * 1024):
        self.app_engine = ApplicationManager()
        self.buffer_size = buffer_size
        self.server_addr = server_address
        self.socket_TCP = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket_TCP.bind(server_address)
        self.socket_TCP.listen(1)
        self.payload_size = struct.calcsize("L")
        self.run()

    def run(self):
        print("TCP server is running on {}".format(self.server_addr))
        while True:
            connection, client_address = self.socket_TCP.accept()
            print("Receive the connection from {}".format(client_address))
            while True:
                # handle client disconnection

                # load the img from the socket
                data = b''
                while len(data) < self.payload_size:
                    tmp = connection.recv(self.buffer_size)
                    if not tmp:
                        print('disconnect')
                        break
                    data += tmp
                packed_msg_size = data[:self.payload_size]
                data = data[self.payload_size:]
                msg_size = struct.unpack("L", packed_msg_size)[0]
                while len(data) < msg_size:
                    data += connection.recv(self.buffer_size)
                frame_data = data[:msg_size]

                im = pickle.loads(frame_data)
                print('[server] receive data at ', datetime.datetime.now())
                t1 = datetime.datetime.now()
                s = time.time()
                bbox = self.app_engine.run(im)
                total = time.time() - s
                print('[server slow compute in ', datetime.datetime.now() - t1)
                # print('ready to send out distributed results')
                print('[server] finish comp at ', datetime.datetime.now())
                data = pickle.dumps([bbox, total])
                print('[server] dumps at ', datetime.datetime.now())
                # data = ujson.dumps([mask,unit])
                data = struct.pack("L", len(data)) + data
                print('[server] packs at ', datetime.datetime.now())
                print('[server] send data at ', datetime.datetime.now())
                connection.sendall(data)

    def signal_handler(self):
        print('receiving close signal.. shutdown socket')
        self.socket_TCP.shutdown(socket.SHUT_RDWR)
        self.socket_TCP.close()


def main():

    from config import Config
    config = Config()
    server = Server(config.servers[1])
    server.run()


if __name__ == "__main__":
    main()
