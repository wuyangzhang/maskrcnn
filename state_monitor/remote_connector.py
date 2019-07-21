import socket
import pickle
import struct
import time
import signal
import datetime


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

    def send(self, data, response, service_id):

        # send requests
        s1 = time.time()
        data = pickle.dumps(data)
        s2 = time.time()
        #print('data dumps in {}'.format(s2-s1))
        data = struct.pack("L", len(data)) + data
        s1 = time.time()
        #print('data pack in {}'.format(s1-s2))

        print('[client] send data {} in {}'.format(service_id, datetime.datetime.now()))

        self.socket_TCP.sendall(data)
        s2 = time.time()
        #print('send data in {}'.format(s2-s1))

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

        s1 = time.time()
        bbox = pickle.loads(res)
        print('[client] receive data {} in {}'.format(service_id, datetime.datetime.now()))

        #print('load pickle in {}'.format(time.time()-s1))
        #print('receive data in {} seconds'.format(time.time() - start))
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
        # s1 = time.time()
        # bbox = pickle.loads(res)
        # print('load pickle in {}'.format(time.time()-s1))
        # print('receive data in {}'.format(time.time() - start))
        # response.append(bbox)


    def disconnect(self):
        self.socket_TCP.shutdown(socket.SHUT_RDWR)
        self.socket_TCP.close()

    def signal_handler(self):
        print('receiving close signal.. shutdown socket')
        self.disconnect()