import time
from typing import List

from lattica import Lattica

from backend.server.rpc_connection_handler import RPCConnectionHandler
from backend.server.static_config import get_model_info
from scheduling.node import RequestSignal
from scheduling.scheduler import Scheduler


class SchedulerManage:
    def __init__(
        self,
        initial_peers: List[str] = [],
        relay_servers: List[str] = [],
        dht_prefix: str = "gradient",
        host_maddrs: List[str] = [],
        announce_maddrs: List[str] = [],
    ):
        self.initial_peers = initial_peers
        self.relay_servers = relay_servers
        self.dht_prefix = dht_prefix
        self.host_maddrs = host_maddrs
        self.announce_maddrs = announce_maddrs

        self.scheduler = None
        self.node_id = f"{dht_prefix}_announce"
        self.lattica = None
        self.stubs = {}

    def run(self, model_name, init_nodes_num):
        self._start_scheduler(model_name, init_nodes_num)
        self._start_lattica()

    def _start_scheduler(self, model_name, init_nodes_num):
        if self.scheduler is not None:
            return

        mode_info = get_model_info(model_name)
        # 初始化 scheduler
        self.scheduler = Scheduler(mode_info, [], min_nodes_bootstrapping=init_nodes_num)

    def _start_lattica(self):
        self.lattica = Lattica.builder().with_listen_addrs(self.host_maddrs)

        if len(self.relay_servers) > 0:
            print(f"Using relay servers: {self.relay_servers}")
            self.lattica.with_relay_servers(self.relay_servers).with_dcutr(True)

        if len(self.announce_maddrs) > 0:
            print(f"Using announce maddrs: {self.announce_maddrs}")
            self.lattica.with_external_addrs(self.announce_maddrs)

        if len(self.initial_peers) > 0:
            print(f"Using initial peers: {self.initial_peers}")
            self.lattica.with_bootstraps(self.initial_peers)

        self.lattica.build()

        self.connection_handler = RPCConnectionHandler(
            lattica=self.lattica,
            scheduler=self.scheduler,
        )

    def get_routing_table(self, request_id, received_ts):
        request = RequestSignal(request_id, received_ts)
        self.scheduler.receive_request(request)

        # 等待最长 5s
        start_time = time.time()
        while request.routing_table is None and (time.time() - start_time) < 5.0:
            time.sleep(0.05)

        # 返回routing_table
        return request.routing_table

    def get_schedule_status(self):
        if self.scheduler is None:
            return "waiting"

        if self.scheduler.layer_allocator.has_full_pipeline():
            return "success"
        else:
            return "waiting"

    def get_call_url_by_node_id(self, node_id):
        return self.connection_handler.get_call_url_by_node_id(node_id)
