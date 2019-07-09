import socket
import sys
import time
import struct

host = 'localhost'
port = 8888
buffersize = 1024
N = 1000000
server_address = (host, port)
client_address = (host, port + 1)


def main():
    if len(sys.argv) < 3:
        usage()
    if sys.argv[1] == '-s' and sys.argv[2] == '-t':
        start_TCP_server()
    if sys.argv[1] == '-s' and sys.argv[2] == '-u':
        start_UDP_server()
    if sys.argv[1] == '-c' and sys.argv[2] == '-t':
        benchmark_TCP(N)
    if sys.argv[1] == '-c' and sys.argv[2] == '-u':
        benchmark_UDP(N)
    else:
        print(sys.argv)
        usage()


def usage():
    sys.stdout = sys.stderr
    print('Usage: echo [-s-c] [-t-u]')
    print('-s: start server')
    print('-c: start client')
    print('-u: use UDP')
    print('-t: use TCP')
    sys.exit(2)


def start_UDP_server():
    socket_UDP_in = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    socket_UDP_in.bind(server_address)

    socket_UDP_out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    socket_UDP_out.connect(client_address)

    print("UDP server is running...")

    while True:
        data = socket_UDP_in.recv(buffersize)
        if not data: break
        socket_UDP_out.sendall(data)
    socket_UDP_out.close()


def start_TCP_server():
    socket_TCP = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    socket_TCP.bind(server_address)
    socket_TCP.listen(1)

    print("TCP server is running...")

    while True:
        connection, client_address = socket_TCP.accept()

        while True:
            data = connection.recv(buffersize)
            if not data: break
            connection.sendall(data)


def benchmark_UDP(N):
    socket_UDP_out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    socket_UDP_out.connect(server_address)

    socket_UDP_in = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    socket_UDP_in.bind(client_address)

    print("Benchmark UDP...")

    duration = 0.0
    for i in range(0, N):
        b = bytes("a" * buffersize, "utf-8")
        start = time.time()
        socket_UDP_out.sendall(b)
        data = socket_UDP_in.recv(buffersize)
        duration += time.time() - start

        if data != b:
            print("Error: Not the same.", data, b)

    print(duration * pow(10, 6) / N, "us for UDP")
    socket_UDP_out.close()


def benchmark_TCP(N, use_TCP_NODELAY=0):
    socket_TCP = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    socket_TCP.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, use_TCP_NODELAY)
    socket_TCP.connect(server_address)

    print("Benchmark TCP...")

    duration = 0.0
    for i in range(0, N):
        start = time.time()
        b = bytes("a" * buffersize, "utf-8")
        socket_TCP.sendall(b)
        data = socket_TCP.recv(buffersize)
        duration += time.time() - start
        if data != b:
            print("Error: Not the same.")

    print(duration * pow(10, 6) / N, "us for TCP")
    socket_TCP.close()


main()