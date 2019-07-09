import socket
import pickle
import struct
import cv2
import time
import ujson

class Client():
    def __init__(self, server_addr, port):
        self.server_addr, self.port = server_addr, port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket_TCP = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket_TCP.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 0)
        self.socket_TCP.connect((self.server_addr, self.port))
        self.buffer_size = 1024 * 1024 * 2
        self.payload_size = struct.calcsize("L")
        self.latency = []

    def run(self):
        data = cv2.imread('./dog.jpg')

        s = time.time()
        start = time.time()

        data = pickle.dumps(data)
        # data = ujson.dumps(data)

        data = struct.pack("L", len(data)) + data
        print('pack latency {}'.format(time.time()-start))
        self.socket_TCP.sendall(data)

        data = b''
        while len(data) < self.payload_size:
            data += self.socket_TCP.recv(self.buffer_size)
        packed_msg_size = data[:self.payload_size]
        data = data[self.payload_size:]


        msg_size = struct.unpack("L", packed_msg_size)[0]

        while len(data) < msg_size:
            data += self.socket_TCP.recv(self.buffer_size)
        res = data[:msg_size]
        start = time.time()
        res = pickle.loads(res)
        # res = ujson.loads(res)
        print('unpack latency {}'.format(time.time() - start))
        print('e2e latency {}'.format(time.time() - s))
        self.latency.append(time.time()-s)

client = Client('127.0.0.1', 5050)

for i in range(100):
    client.run()

print(sum(client.latency) / len(client.latency))