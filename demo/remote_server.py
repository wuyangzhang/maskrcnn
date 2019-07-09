import socket
import pickle
import struct
import ujson

from mobile_client import MaskCompute

import time

class Server():

    def __init__(self, server_address, buffer_size=1024 * 1024):
        self.mask_engine = MaskCompute()
        self.buffer_size = buffer_size
        self.socket_TCP = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket_TCP.bind(server_address)
        self.socket_TCP.listen(1)
        self.payload_size = struct.calcsize("L")

    def run(self):
        print("TCP server is running...")
        while True:
            connection, client_address = self.socket_TCP.accept()
            print("Receive the connection from {}".format(client_address))
            while True:
                # load the img from the socket
                data = b''
                while len(data) < self.payload_size:
                    data += connection.recv(self.buffer_size)
                packed_msg_size = data[:self.payload_size]
                data = data[self.payload_size:]
                msg_size = struct.unpack("L", packed_msg_size)[0]
                while len(data) < msg_size:
                    data += connection.recv(self.buffer_size)
                frame_data = data[:msg_size]

                im = pickle.loads(frame_data)
                # im = ujson(frame_data)
                print('receive frame data in the shape {}'.format(im.shape))
                _, mask, unit = self.mask_engine.run(im)

                print('ready to send out mask')
                data = pickle.dumps([mask,unit])
                # data = ujson.dumps([mask,unit])
                data = struct.pack("L", len(data)) + data
                connection.sendall(data)


def main():
    server = Server(('127.0.0.1', 5050))
    server.run()


if __name__ == "__main__":
    main()
