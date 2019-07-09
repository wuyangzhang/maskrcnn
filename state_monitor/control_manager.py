import collections
import queue
import time
import socket
import pickle
import struct

class RemoteServer():
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

    def send(self, data):
        data = pickle.dumps(data)
        data = struct.pack("L", len(data)) + data
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
        mask, unit = pickle.loads(res)
        print('receive data in {}'.format(time.time()-start))
        return mask, unit

    def disconnect(self):
        self.socket_TCP.close()

class ControlManager():

    def __init__(self, prediction_mgr, partition_mgr, mask_engine, server_conf):
        self.branch_states = {0: 'cold_start', 1: 'distribute', 2: 'shortcut'}
        self.curr_state = 0
        self.prediction_mgr = prediction_mgr
        self.partition_mgr = partition_mgr
        self.mask_engine = mask_engine
        self.time_counter = collections.defaultdict(list)
        self.distributed_res = list()
        self.remote_servers = dict()
        self.init_remote_servers(server_conf)
        self.last_composite = None
        self.use_local = True
        self.local_composite = None

    def init_remote_servers(self, server_conf):
        with open(server_conf) as f:
            for line in f.readlines():
                id, ip, port = line.split(',')
                self.remote_servers[id] = RemoteServer(id, ip, port)

    def set_curr_state(self, state):
        if state not in self.branch_states:
            return
        self.curr_state = state

    def get_curr_state(self):
        if self.prediction_mgr.get_queue_len() < self.prediction_mgr.max_queue_size:
            return self.branch_states[0]
        elif self.prediction_mgr.get_queue_len() >= self.prediction_mgr.max_queue_size:
            return self.branch_states[1]


    def distribute(self, partitions):
        if self.use_local:
            local_partition = partitions[0]
            self.local_composite, bbox, units = self.mask_engine.run(local_partition)
            self.distributed_res.append((bbox, units))

            #todo[Priority Highest]: should do in parallel!!!!
            for i, id in enumerate(self.remote_servers):
                self.time_counter[i].append(time.time())
                bbox, units = self.remote_servers[id].send(partitions[i+1])
                self.time_counter[i] = time.time() - self.time_counter[i][-1]
                self.distributed_res.append((bbox, units))
        else:
            for i, id in enumerate(self.remote_servers):
                self.time_counter[i].append(time.time())
                bbox, units = self.remote_servers[id].send(partitions[i])
                self.time_counter[i] = time.time() - self.time_counter[i][-1]
                self.distributed_res.append((bbox, units))

        #todo: thread join!


    """ Merge Partitions

    this function takes as the input the historical e2e latency in order to 
    evaluate the computation capability of all the involved nodes.

    Args:  
        None   

    Returns:  
        N partitions  

    """
    def merge_partitions(self):
        self.partition_mgr.merge_partition(self.distributed_res)
        return self.local_composite, 0, 0


    """ Resource report.   

    this function takes as the input the historical e2e latency in order to 
    evaluate the computation capability of all the involved nodes.
    
    Args:  
        None   
             
    Returns:  
        N partitions  

    """
    def report_resources(self):
        count = len(self.remote_servers)
        if self.use_local:
            count += 1
        return [0.2] * count

        capability = []
        for id in self.time_counter:
            capability.append(self.time_counter[id])
        return capability



