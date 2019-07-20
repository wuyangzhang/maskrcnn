import socket
import pickle
import struct
import time
import signal


class RemoteConnector:
    def __init__(self, id, ip, port):
        self.id = id
        self.ip = ip
        self.port = int(port)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.socket_TCP = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket_TCP.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 0)
        self.socket_TCP.connect((ip, self.port))
        self.buffer_size = 1024 * 1024
        self.payload_size = struct.calcsize("L")
        self.e2e_latency = []
        signal.signal(signal.SIGINT, self.signal_handler)

    def get_avg_latency(self):
        return sum(self.e2e_latency) / len(self.e2e_latency)

    def send(self, data, response):

        # send requests
        data = pickle.dumps(data)
        data = struct.pack("L", len(data)) + data
        print('send data to remote servers')
        self.socket_TCP.sendall(data)


        start = time.time()
        data = b''
        while len(data) < self.payload_size:
            data += self.socket_TCP.recv(self.buffer_size)
        packed_msg_size = data[:self.payload_size]
        data = data[self.payload_size:]
        msg_size = struct.unpack("L", packed_msg_size)[0]
        while len(data) < msg_size:
            data += self.socket_TCP.recv(self.buffer_size)
        res = data[:msg_size]
        bbox = pickle.loads(res)
        print('receive data in {} seconds'.format(time.time() - start))
        response.append(bbox)

        # data = []
        # while len(data) < self.payload_size:
        #     data.append(self.socket_TCP.recv(self.buffer_size))
        # packed_msg_size = data[:self.payload_size]
        # data = data[self.payload_size:]
        # msg_size = struct.unpack("L", packed_msg_size)[0]
        # while len(data) < msg_size:
        #     data.append(self.socket_TCP.recv(self.buffer_size))
        # res = data[:msg_size]
        # mask, unit = pickle.loads(res)
        # print('receive data in {}'.format(time.time() - start))
        # response.append(mask)
        # response.append(unit)

    def disconnect(self):
        self.socket_TCP.shutdown(socket.SHUT_RDWR)
        self.socket_TCP.close()

    def signal_handler(self):
        print('receiving close signal.. shutdown socket')
        self.disconnect()