import json
import time

from lattica import ConnectionHandler, Lattica, rpc_method

from backend.server.static_config import get_model_info
from parallax_utils.logging_config import get_logger
from scheduling.node import Node, NodeHardwareInfo
from scheduling.scheduler import Scheduler

logger = get_logger(__name__)


class RPCConnectionHandler(ConnectionHandler):
    """
    Handles RPC requests from clients, forwarding them to the appropriate TransformerBackend.
    Inherits from hivemind's ConnectionHandler.
    """

    def __init__(
        self,
        lattica: Lattica,
        scheduler: Scheduler,
    ):
        # Initialize the base class
        super().__init__(lattica)
        self.scheduler = scheduler
        self.call_url_map = {}

    @rpc_method
    def node_join(self, message):
        print(f"node_join: {message}")
        node = json.loads(message)
        # node = {
        #     "call_url": "http://127.0.0.1:8000",
        #     "node_id": "lattica peer id",
        #     "hardware": {
        #         "node_id": "lattica peer id",
        #         "tflops_fp16": 100,
        #         "memory_gb": 100,
        #         "memory_bandwidth_gbps": 100,
        #     },
        #     "model_name": "",
        #     "kv_cache_ratio": 0.3,
        #     "param_hosting_ratio": 0.5,
        #     "max_concurrent_requests": 16,
        #     "max_sequence_length": 1024,
        # }
        node = self.build_node(node)
        self.scheduler.enqueue_join(node)

        self.call_url_map[node.get("node_id")] = node.get("call_url")

        # 等待 layer 分配完成
        return self.wait_layer_allocation(node.get("node_id"), wait_seconds=300)

    @rpc_method
    def node_leave(self, message):
        print(f"node_leave: {message}")
        node = json.loads(message)
        node_id = node.get("node_id")
        self.scheduler.enqueue_leave(node_id)
        return {}

    @rpc_method
    def node_update(self, message):
        print(f"node_update: {message}")
        node = json.loads(message)
        self.scheduler.enqueue_node_update(node)

        self.call_url_map[node.get("node_id")] = node.get("call_url")

        return {}

    def wait_layer_allocation(self, current_node_id, wait_seconds):
        start_time = time.time()
        while True:
            layer_allocation = self.get_layer_allocation(current_node_id)
            if layer_allocation:
                return layer_allocation
            if time.time() - start_time > wait_seconds:
                return {}
            time.sleep(0.5)

    def get_layer_allocation(self, current_node_id):
        list_node_allocations = self.scheduler.list_node_allocations()
        # 判断有哪些需要更新
        for node_id, start_layer, end_layer in list_node_allocations:
            if current_node_id == node_id:
                return {"node_id": node_id, "start_layer": start_layer, "end_layer": end_layer}
        return {}

    def build_node(self, node_json):
        return Node(
            node_id=node_json.get("node_id"),
            hardware=self.build_hardware(node_json.get("hardware")),
            model_info=get_model_info(node_json.get("model_name")),
            kv_cache_ratio=node_json.get("kv_cache_ratio"),
            param_hosting_ratio=node_json.get("param_hosting_ratio"),
            max_concurrent_requests=node_json.get("max_concurrent_requests"),
            max_sequence_length=node_json.get("max_sequence_length"),
        )

    def build_hardware(self, hardware_json):
        node_id = hardware_json.get("node_id")
        tflops_fp16 = hardware_json.get("tflops_fp16")
        memory_gb = hardware_json.get("memory_gb")
        memory_bandwidth_gbps = hardware_json.get("memory_bandwidth_gbps")
        return NodeHardwareInfo(
            node_id=node_id,
            tflops_fp16=tflops_fp16,
            memory_gb=memory_gb,
            memory_bandwidth_gbps=memory_bandwidth_gbps,
        )

    def get_call_url_by_node_id(self, node_id):
        return self.call_url_map.get(node_id, None)
