import collections
import time
from state_monitor.remote_server import RemoteServer


class ControlManager:

    def __init__(self, config, prediction_mgr, partition_mgr, mask_engine):

        self.config = config
        self.branch_states = {0: 'cache_refresh', 1: 'distribute', 2: 'shortcut'}
        self.curr_state = 0
        self.prediction_mgr = prediction_mgr
        self.partition_mgr = partition_mgr
        self.mask_engine = mask_engine
        self.time_counter = collections.defaultdict(list)
        self.distributed_res = list()
        self.remote_servers = dict()
        # self.init_remote_servers(server_conf)
        self.last_composite = None
        self.use_local = True
        self.local_composite = None

    def init_remote_servers(self, server_conf):
        with open(server_conf) as f:
            for line in f.readlines():
                id, ip, port = line.split(',')
                self.remote_servers[id] = RemoteServer(id, ip, port)

    def set_branch_state(self, state):
        if state not in self.branch_states:
            return
        self.curr_state = state

    def get_branch_state(self):
        if not self.prediction_mgr.is_active():
            return self.branch_states[0]
        elif self.prediction_mgr.is_active():
            return self.branch_states[1]

    def dist_jobs(self, partitions):
        if self.use_local:
            local_partition = partitions[0]
            self.local_composite, bbox, units = self.mask_engine.run(local_partition)
            self.distributed_res.append((bbox, units))

            # todo[Priority Highest]: should do in parallel!!!!
            for i, id in enumerate(self.remote_servers):
                self.time_counter[i].append(time.time())
                bbox, units = self.remote_servers[id].send(partitions[i + 1])
                self.time_counter[i] = time.time() - self.time_counter[i][-1]
                self.distributed_res.append((bbox, units))
        else:
            for i, id in enumerate(self.remote_servers):
                self.time_counter[i].append(time.time())
                bbox, units = self.remote_servers[id].send(partitions[i])
                self.time_counter[i] = time.time() - self.time_counter[i][-1]
                self.distributed_res.append((bbox, units))

        # todo: thread join!

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
