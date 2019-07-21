import collections
import threading

from state_monitor.remote_connector import RemoteConnector


class ControlManager:

    def __init__(self, config, prediction_mgr, partition_mgr, mask_engine):

        self.prediction_mgr = prediction_mgr
        self.partition_mgr = partition_mgr
        self.mask_engine = mask_engine

        self.config = config
        self.branch_states = {0: 'cache_refresh', 1: 'distribute', 2: 'shortcut'}
        self.curr_state = 0

        self.time_counter = collections.defaultdict(list)
        self.distributed_res = list()

        self.total_remote_servers = config.total_remote_servers
        self.remote_servers = config.servers
        self.sockets = dict()
        self.init_remote_servers()

        self.service_id = 0
        # self.use_local = config.use_local
        # self.local_composite = None

    def clean_cache(self):
        self.time_counter = collections.defaultdict(list)
        self.distributed_res.clear()

    def init_remote_servers(self):
        """
        create remote sockets
        :return remote sockets
        """
        for id in self.remote_servers:
            ip, port = self.remote_servers[id]
            self.sockets[id] = RemoteConnector(id, ip, port)

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
        self.service_id += 1
        threads = [None] * self.total_remote_servers
        #self.distributed_res = [[]] * self.total_remote_servers
        self.distributed_res = collections.defaultdict(list)

        #print('distribute the workloads to the servers..')
        for i, sid in enumerate(self.sockets.keys()):
            #self.time_counter[i].append(time.time())

            threads[i] = threading.Thread(target=self.sockets[sid].send,
                                          args=(partitions[i], self.distributed_res[sid], self.service_id))

            threads[i].start()
            #self.time_counter[i] = time.time() - self.time_counter[i][-1]


        for id, thread in enumerate(threads):
            thread.join()

        #print('All distributed results are ready.')
        # if self.use_local:
        #     local_partition = partitions[0]
        #     #self.local_composite, bbox, units = self.mask_engine.run(local_partition)
        #     threads[0] = threading.Thread(target=self.mask_engine.run,
        #                                   args=(local_partition, results[0]))
        #     threads[0].start()
        #
        #     for i, id in enumerate(self.remote_servers):
        #         self.time_counter[i].append(time.time())
        #
        #         threads[i+1] = threading.Thread(target = self.remote_servers[id].send,
        #                                         args=(partitions[i+1]))
        #         threads[i+1].start()
        #         #bbox, units = self.remote_servers[id].send(partitions[i + 1])
        #         self.time_counter[i] = time.time() - self.time_counter[i][-1]
        #         #self.distributed_res.append((bbox, units))
        # else:
        #     for i, id in enumerate(self.remote_servers):
        #         self.time_counter[i].append(time.time())
        #         threads[i] = threading.Thread(target=self.remote_servers[id].send,
        #                                       args=(partitions[i], results[i]))
        #         # bbox, units = self.remote_servers[id].send(partitions[i])
        #         # self.time_counter[i] = time.time() - self.time_counter[i][-1]
        #         # self.distributed_res.append((bbox, units))

    # def total_comp_devices(self):
    #     return self.total_remote_servers if not self.use_local else self.total_remote_servers + 1

    def merge_partitions(self):
        """ Merge Partitions

        this function takes as the input the historical e2e latency in order to
        evaluate the computation capability of all the involved nodes.

        :param None
        :return N partitions
        """
        return self.partition_mgr.merge_partition(self.distributed_res)

    def report_resources(self):
        """ Resource report.

        this function takes as the input the historical e2e latency in order to
        evaluate the computation capability of all the involved nodes.

        :param None
        :return N partitions
        """

        # todo figure out the condition
        if True:
            return [0.2] * self.total_remote_servers

        # todo : load historical time
        capability = []
        for id in self.time_counter:
            capability.append(self.time_counter[id])
        return capability
