import argparse
import math
import dataclasses
import ipaddress
import subprocess
from collections import defaultdict
from enum import IntEnum
from pathlib import Path
from typing import Optional, Dict, List, TextIO, Set
import yaml
from mininet.cli import CLI
from mininet.link import TCLink
from mininet.net import Mininet
from mininet.node import Node, OVSBridge
from mininet.topo import Topo


def validate_definition_file(definition_file: Path):
    if not definition_file.exists():
        raise FileNotFoundError(definition_file)
    if not definition_file.is_file():
        raise FileNotFoundError(definition_file)
    if not definition_file.suffix == '.yaml':
        raise ValueError(f'File {definition_file} is not a yaml')
    return


def read_definition(definition_file: Path) -> dict:
    with open(definition_file, 'r') as def_file:
        return yaml.safe_load(def_file)


def get_subnet(address: str, mask: str) -> str:
    network = ipaddress.IPv4Network(f"{address}/{mask}", strict=False)
    return str(network.network_address)


# def get_subnet(ip: str, mask: str) -> str:
#     split_ip = ip.split(".")
#     split_mask = mask.split(".")
#     for i in range(len(split_mask)):
#         split_ip[i] = str(int(split_ip[i]) & int(split_mask[i]))
#     return ".".join(split_ip)


def dotted_to_mask(mask_dotted: str) -> int:
    split_mask = [bin(int(el)) for el in mask_dotted.split('.')]
    return sum(el.count('1') for el in split_mask)


class NodeType(IntEnum):
    ROUTER = 1
    HOST = 2


@dataclasses.dataclass(eq=False)
class NodeDefinition:
    address: str
    mask: str
    link_name: str
    node_name: str
    node_type: NodeType

    def complete_address(self):
        return f"{get_subnet(self.address, self.mask)}/{dotted_to_mask(self.mask)}"


@dataclasses.dataclass
class FlowDemand:
    source_node: str
    destination_node: str
    rate: float


class LinuxRouter(Node):
    def config(self, **params):
        super().config(**params)
        self.cmd("sysctl net.ipv4.ip_forward=1")

    def terminate(self):
        self.cmd("sysctl net.ipv4.ip_forward=0")
        super().terminate()


class NetworkTopology(Topo):
    def __init__(self, subnet_to_nodes):
        self._subnet_to_nodes = subnet_to_nodes
        self.switch_id: int = 0
        super().__init__()

    def build(self):
        # create nodes
        nodes = [n for v in self._subnet_to_nodes.values() for n in v]
        hosts: List[NodeDefinition] = [n for n in nodes if n.node_type == NodeType.HOST]
        for host in hosts:
            host_subnet = get_subnet(host.address, host.mask)
            subnet_nodes = self._subnet_to_nodes[host_subnet]
            router = [n for n in subnet_nodes if n.node_type == NodeType.ROUTER][0]
            self.addHost(host.node_name, ip=host.address, defaultRoute=f"via {router.address}")

        # create routers
        router_names = [n.node_name for n in nodes if n.node_type == NodeType.ROUTER]
        router_names = list(set(router_names))
        for router_name in router_names:
            self.addNode(router_name, cls=LinuxRouter, ip=None)

        # link stuff together
        for subnet, subnet_nodes in self._subnet_to_nodes.items():
            if len(subnet_nodes) == 2:
                node_1 = subnet_nodes[0]
                node_2 = subnet_nodes[1]
                intfname1 = f"{node_1.node_name}-{node_1.link_name}-{node_2.node_name}-{node_2.link_name}"
                intfname2 = f"{node_2.node_name}-{node_2.link_name}-{node_1.node_name}-{node_1.link_name}"
                self.addLink(node_1.node_name, node_2.node_name, cls=TCLink,
                             intfName1=intfname1, params1={"ip": f"{node_1.address}/{dotted_to_mask(node_1.mask)}"},
                             intfName2=intfname2, params2={"ip": f"{node_2.address}/{dotted_to_mask(node_2.mask)}"})
            else:
                # create switch  and connect everything to the switch
                switch_name = f"switch{self.switch_id}"
                switch = self.addSwitch(switch_name)
                self.switch_id += 1
                for node in subnet_nodes:
                    intfname2 = f"{node.node_name}-{node.link_name}-{switch_name}"
                    self.addLink(switch, node.node_name,
                                 intfName2=intfname2, params2={"ip": f"{node.address}/{dotted_to_mask(node.mask)}"})
                pass
        pass


class NetworkDefinition:
    def __init__(self, network_definition: dict):
        self._subnet_to_nodes: Dict[str, List[NodeDefinition]] = defaultdict(list)
        self._subnet_to_cost: Dict[str, int] = defaultdict(lambda: 1)
        routers: dict = network_definition.get("routers")
        hosts: dict = network_definition.get("hosts")
        demands: List[dict] = network_definition.get("demands")
        self._flow_demands: List[FlowDemand] = [FlowDemand(source_node=demand.get("src"),
                                                           destination_node=demand.get("dst"),
                                                           rate=demand.get("rate")) for demand in demands]
        self._flow_to_goodput: Dict[int, float] = {}
        self._flow_to_routes: Dict[int, List[str]] = defaultdict(list)
        self._load_routers(routers_def=routers)
        self._load_hosts(hosts_def=hosts)
        self._create_cplex_file()
        self._solve_linear_program()
        self._get_flow_routes()

    def _get_flow_routes(self):
        lines: List[str]
        with open(self._get_output_file(), "r") as of:
            lines = of.readlines()
        lines = [' '.join(line.split()) for line in lines]
        lines = [line for line in lines if line != ""]
        filtered_lines: List[str] = []
        for line in lines:
            elements = line.split(" ")
            if not elements:
                continue
            if not elements[0].isdigit():
                continue
            filtered_lines.append(line)
        for line in filtered_lines:
            columns = line.split(" ")
            var_name: str = columns[1]
            if var_name.startswith("lambda_"):
                flow_idx = int(var_name.lstrip("lambda_"))
                self._flow_to_goodput[flow_idx] = float(columns[2])
            if var_name.startswith("fbr"):
                flow_idx, link = var_name.lstrip("fbr").split("_", 1)
                value: float = float(columns[2])
                if value != 0:
                    self._flow_to_routes[int(flow_idx)].append(link)
        return

    def _solve_linear_program(self):
        subprocess.run(["glpsol", "--lp", self._get_cplex_file(), "-o", self._get_output_file()],
                       stdout=subprocess.DEVNULL)
        return

    def _get_router_connected_to_host(self, host_name: str) -> Optional[str]:
        for subnet, nodes in self._subnet_to_nodes.items():
            node_names: List[str] = [node.node_name for node in nodes]
            if host_name in node_names:
                return [node.node_name for node in nodes if node.node_type == NodeType.ROUTER][0]
        return None

    def _get_adjacent_routers(self, router_name: str) -> Set[str]:
        adjacent_routers: Set[str] = set()
        for subnet, nodes in self._subnet_to_nodes.items():
            node_names: List[str] = [n.node_name for n in nodes if n.node_type == NodeType.ROUTER]
            if router_name in node_names:
                adj: List[str] = [name for name in node_names if name != router_name]
                for node in adj:
                    adjacent_routers.add(node)
        return adjacent_routers

    @staticmethod
    def _get_output_folder() -> Path:
        return Path(__file__).parent.parent.joinpath("TMP")

    @staticmethod
    def _get_cplex_file() -> Path:
        return NetworkDefinition._get_output_folder().joinpath("cplex_definition.lp")

    @staticmethod
    def _get_output_file() -> Path:
        return NetworkDefinition._get_output_folder().joinpath("sol.txt")

    def _create_cplex_file(self):
        output_folder: Path = self._get_output_folder()
        output_folder.mkdir(exist_ok=True)
        cplex_definition_file: Path = self._get_cplex_file()
        if cplex_definition_file.is_file():
            cplex_definition_file.unlink()
        cplex_definition_file.touch()

        cplex_def: TextIO = open(cplex_definition_file, "w")
        denominator: float = math.prod([fd.rate for fd in self._flow_demands])

        objective_function: str = "obj: rate_min + D\n"

        cplex_def.writelines(["Maximize\n", objective_function, "\n"])

        # Subject to Section
        subject_to: List[str] = ["Subject to"]
        # CPLEX is sensitive to spaces, we must make sure to add the right indentation
        indentation: str = " "*4
        # definition of minimal effectiveness ratio
        subject_to.append(f"{indentation}\\ rate_min is the minimum of all the flow rates")
        for idx, demand in enumerate(self._flow_demands):
            subject_to.append(f"{indentation}rate_min_{idx}: rate_{idx} - rate_min >= 0")

        for idx, demand in enumerate(self._flow_demands):
            subject_to.append(f"{indentation}ratio_{idx}: lambda_{idx} - {demand.rate} rate_{idx} = 0")

        for idx, demand in enumerate(self._flow_demands):
            subject_to.append(f"{indentation}limit_{idx}: rate_{idx} <= 1")

        # sum of ratio divided by the product of the desired rates
        sum_ratio: str = f"{indentation}sum_ratio: "
        sum_ratio += " + ".join([f"rate_{i}" for i in range(len(self._flow_demands))])
        sum_ratio += f" - {denominator} D = 0"
        subject_to.append(sum_ratio)

        # flow balance for real-value rate variables
        subject_to.append(f"{indentation}\\ flow balance for real-value rate variables")
        router_names: Set[str] = {n.node_name for v in self._subnet_to_nodes.values()
                                  for n in v if n.node_type == NodeType.ROUTER}
        for idx, flow_demand in enumerate(self._flow_demands):
            source_router: str = self._get_router_connected_to_host(flow_demand.source_node)
            destination_router: str = self._get_router_connected_to_host(flow_demand.destination_node)
            for router_name in router_names:
                adjacent_routers = self._get_adjacent_routers(router_name=router_name)
                flow_balance = f"{indentation}flow_{router_name}_br{idx}: "
                tmp = []
                for adjacent_router in adjacent_routers:
                    tmp.append(f"fbr{idx}_{adjacent_router}_{router_name} - fbr{idx}_{router_name}_{adjacent_router}")
                flow_balance += " + ".join(tmp)
                if router_name == source_router:
                    flow_balance += f" + lambda_{idx}"
                if router_name == destination_router:
                    flow_balance += f" - lambda_{idx}"
                flow_balance += " = 0"
                subject_to.append(flow_balance)
            # Used to visually separate flow for easier inspection
            subject_to.append("")

        # flow balance for indicators, for each flow, for each node
        subject_to.append(f"{indentation}\\ flow balance for indicators, for each flow, for each node")
        for idx, flow_demand in enumerate(self._flow_demands):
            source_router: str = self._get_router_connected_to_host(flow_demand.source_node)
            destination_router: str = self._get_router_connected_to_host(flow_demand.destination_node)
            for router_name in router_names:
                adjacent_routers = self._get_adjacent_routers(router_name=router_name)
                flow_balance = f"{indentation}flow_{router_name}_b{idx}: "
                tmp = []
                for adjacent_router in adjacent_routers:
                    tmp.append(f"fbi{idx}_{adjacent_router}_{router_name} - fbi{idx}_{router_name}_{adjacent_router}")
                flow_balance += " + ".join(tmp)
                if router_name == source_router:
                    flow_balance += f" = - 1"
                elif router_name == destination_router:
                    flow_balance += f" = 1"
                else:
                    flow_balance += " = 0"
                subject_to.append(flow_balance)
            # Used to visually separate flow for easier inspection
            subject_to.append("")

        subject_to.extend(self._compute_cplex_in_out_flow())

        # link capacities, for each link
        link_capacities: List[str] = self._compute_cplex_link_capacities()
        subject_to.extend(link_capacities)

        # control of real-value flow variables by corresponding indicators, for each flow and link
        real_indicators: List[str] = self._compute_cplex_real_value_indicators_control()
        subject_to.extend(real_indicators)

        # Binary section
        binaries: List[str] = ["Binary"]
        binaries.extend(self._compute_cplex_binaries())

        subject_to = [f"{line}\n" for line in subject_to]
        binaries = [f"{line}\n" for line in binaries]
        cplex_def.writelines(subject_to)
        cplex_def.writelines(binaries)
        cplex_def.write("End")
        cplex_def.close()
        return

    def _compute_cplex_in_out_flow(self) -> List[str]:
        incoming: List[str] = []
        outgoing: List[str] = []
        indentation: str = " "*4
        router_names: Set[str] = {n.node_name for v in self._subnet_to_nodes.values()
                                  for n in v if n.node_type == NodeType.ROUTER}
        # mutual exclusion of incoming flows into same node, for each flow, for each node
        incoming.append(f"{indentation}\\ mutual exclusion of incoming flows into same node, for each flow, for each node")
        outgoing.append(f"{indentation}\\ mutual exclusion of outgoing out of same node, for each flow, for each node")
        for idx, flow_demand in enumerate(self._flow_demands):
            for router_name in router_names:
                adjacent_routers = self._get_adjacent_routers(router_name=router_name)
                flow_balance_in = f"{indentation}in_{router_name}_{idx}: "
                flow_balance_out = f"{indentation}out_{router_name}_{idx}: "
                tmp_in = []
                tmp_out = []
                for adjacent_router in adjacent_routers:
                    tmp_in.append(f"fbi{idx}_{adjacent_router}_{router_name}")
                    tmp_out.append(f"fbi{idx}_{router_name}_{adjacent_router}")
                flow_balance_in += " + ".join(tmp_in)
                flow_balance_in += " <= 1"
                flow_balance_out += " + ".join(tmp_out)
                flow_balance_out += " <= 1"
                incoming.append(flow_balance_in)
                outgoing.append(flow_balance_out)
            # Used to visually separate flow for easier inspection
            incoming.append("")
            outgoing.append("")
        return incoming + outgoing

    def _compute_cplex_binaries(self) -> List[str]:
        indentation: str = " "*4
        binaries: List[str] = []
        router_names: Set[str] = {n.node_name for v in self._subnet_to_nodes.values()
                                  for n in v if n.node_type == NodeType.ROUTER}
        for idx, flow_demand in enumerate(self._flow_demands):
            for router_name in router_names:
                adjacent_routers = self._get_adjacent_routers(router_name=router_name)
                for adjacent_router in adjacent_routers:
                    binaries.append(f"{indentation}fbi{idx}_{adjacent_router}_{router_name}")
            binaries.append("")
        return binaries

    def _compute_cplex_real_value_indicators_control(self) -> List[str]:
        indentation: str = " " * 4
        real_indicators: List[str] = [f"{indentation}\\ control of real-value flow variables by corresponding indicators, for each flow and link"]
        # given that router are connected with point to point links they must be in a subnet which contain only
        # two routers
        router_links = [(nodes, sub_net) for sub_net, nodes in self._subnet_to_nodes.items()
                        if len(nodes) == 2
                        and nodes[0].node_type == NodeType.ROUTER
                        and nodes[1].node_type == NodeType.ROUTER]
        for idx, flow_demand in enumerate(self._flow_demands):
            for router_link in router_links:
                link_routers = router_link[0]
                subnet = router_link[1]
                cost = self._subnet_to_cost[subnet]
                router_1: NodeDefinition = link_routers[0]
                router_2: NodeDefinition = link_routers[1]
                r1_r2_cx = f"{indentation}{router_1.node_name}_{router_2.node_name}_c{idx}: "
                real = f"fbr{idx}_{router_1.node_name}_{router_2.node_name} - {cost} fbi{idx}_{router_1.node_name}_{router_2.node_name}"
                r1_r2_cx += real
                r1_r2_cx += " <= 0"
                r2_r1_cx = f"{indentation}{router_2.node_name}_{router_1.node_name}_c{idx}: "
                real = f"fbr{idx}_{router_2.node_name}_{router_1.node_name} - {cost} fbi{idx}_{router_2.node_name}_{router_1.node_name}"
                r2_r1_cx += real
                r2_r1_cx += " <= 0"
                real_indicators.append(r1_r2_cx)
                real_indicators.append(r2_r1_cx)
            real_indicators.append("")
        real_indicators.append("")
        return real_indicators

    def _compute_cplex_link_capacities(self) -> List[str]:
        indentation: str = " " * 4
        link_capacities: List[str] = [f"{indentation}\\ link capacities, for each link"]
        # given that router are connected with point to point links they must be in a subnet which contain only
        # two routers
        router_links = [(nodes, sub_net) for sub_net, nodes in self._subnet_to_nodes.items()
                        if len(nodes) == 2
                        and nodes[0].node_type == NodeType.ROUTER
                        and nodes[1].node_type == NodeType.ROUTER]
        for router_link in router_links:
            link_routers = router_link[0]
            subnet = router_link[1]
            router_1: NodeDefinition = link_routers[0]
            router_2: NodeDefinition = link_routers[1]
            link_capacity_name: str = f"{indentation}{router_1.node_name}_{router_2.node_name}_c: "
            tmp = []
            for idx, flow_demand in enumerate(self._flow_demands):
                tmp.append(f"fbr{idx}_{router_1.node_name}_{router_2.node_name}")
                tmp.append(f"fbr{idx}_{router_2.node_name}_{router_1.node_name}")
            link_capacity_name += " + ".join(tmp)
            link_capacity_name += f" <= {self._subnet_to_cost[subnet]}"
            link_capacities.append(link_capacity_name)
        link_capacities.append("")
        return link_capacities

    def _load_routers(self, routers_def: dict):
        for routers_name, routers_def in routers_def.items():
            for link_name, link_def in routers_def.items():
                address: str = link_def.get("address")
                mask: str = link_def.get("mask")
                cost: Optional[int] = link_def.get("cost")
                subnet: str = get_subnet(address, mask)
                if cost is not None:
                    self._subnet_to_cost[subnet] = cost
                node: NodeDefinition = NodeDefinition(address=address, mask=mask,
                                                      node_type=NodeType.ROUTER,
                                                      link_name=link_name, node_name=routers_name)
                self._subnet_to_nodes[subnet].append(node)
        return

    def _load_hosts(self, hosts_def: dict):
        for host_name, hosts_def in hosts_def.items():
            for link_name, link_def in hosts_def.items():
                address: str = link_def.get("address")
                mask: str = link_def.get("mask")
                subnet: str = get_subnet(address, mask)
                cost = 1
                if cost is not None:
                    self._subnet_to_cost[subnet] = cost
                node: NodeDefinition = NodeDefinition(address=address, mask=mask,
                                                      node_type=NodeType.HOST,
                                                      link_name=link_name, node_name=host_name)
                self._subnet_to_nodes[subnet].append(node)
        return

    def _find_shortest_paths(self):
        node_to_paths = {}
        nodes = [n for v in self._subnet_to_nodes.values() for n in v]
        for node in nodes:
            node_to_dist, node_to_prev = self._dijkstra(source_node=node)
            node_to_paths[node.node_name] = (node_to_dist, node_to_prev)
        return node_to_paths

    @staticmethod
    def _find_vertex_with_smallest_distance(Q, router_to_dist):
        dist_to_router = {dist: router for router, dist in router_to_dist.items() if router in Q}
        min_dist = min(dist_to_router.keys())
        return dist_to_router[min_dist]

    def _find_neighbours(self, source_node_name: str):
        neighbours = []
        for subnet, nodes in self._subnet_to_nodes.items():
            node_names = [n.node_name for n in nodes]
            if source_node_name in node_names:
                neighbours.extend([node for node in nodes if node.node_name != source_node_name])
        return neighbours

    def _dijkstra(self, source_node):
        # Dijkstra as seen on wikipedia https://en.wikipedia.org/wiki/Dijkstra's_algorithm
        node_to_dist: Dict[str, float] = {}
        node_to_prev: Dict[str, Optional[str]] = {}
        nodes = [n for v in self._subnet_to_nodes.values() for n in v]
        Q = [n.node_name for n in nodes]
        Q = list(set(Q))
        for node in nodes:
            node_to_dist[node.node_name] = math.inf
            node_to_prev[node.node_name] = None
        node_to_dist[source_node.node_name] = 0
        while len(Q) > 0:
            u = self._find_vertex_with_smallest_distance(Q, node_to_dist)
            Q = [r for r in Q if r != u]

            # find neighbours of u still in Q
            neighbours = self._find_neighbours(u)
            for conn in neighbours:
                cost = self._subnet_to_cost[get_subnet(conn.address, conn.mask)]
                alt = node_to_dist[u] + cost
                v = conn.node_name
                if alt < node_to_dist[v]:
                    node_to_dist[v] = alt
                    node_to_prev[v] = u
        return node_to_dist, node_to_prev

    def find_shortest_link_between(self, node_name1: str, node_name2: str) -> NodeDefinition:
        common_subnets: list[str] = []
        for subnet, nodes in self._subnet_to_nodes.items():
            contained_nodes: list[str] = [n.node_name for n in nodes]
            if node_name1 in contained_nodes and node_name2 in contained_nodes:
                common_subnets.append(subnet)
        cost_to_subnet = {cost: subnet for subnet, cost in self._subnet_to_cost.items() if subnet in common_subnets}
        min_cost = min(cost_to_subnet.keys())
        cheapest_subnet = cost_to_subnet[min_cost]
        # If a node has multiple interfaces in the cheapest subnet I believe that taking any of them should suffice
        return [n for n in self._subnet_to_nodes[cheapest_subnet] if n.node_name == node_name2][0]

    @staticmethod
    def _get_node_definition_in_same_link(router_links, router_name1: str, router_name2: str):
        for nodes, subnet in router_links:
            if nodes[0].node_name == router_name1 and nodes[1].node_name == router_name2:
                return nodes[0], nodes[1]
            if nodes[0].node_name == router_name2 and nodes[1].node_name == router_name1:
                return nodes[1], nodes[0]
        return

    def _get_connected_router(self, host: NodeDefinition):
        subnet = get_subnet(host.address, host.mask)
        nodes = self._subnet_to_nodes[subnet]
        return [n for n in nodes if n.node_type == NodeType.ROUTER][0]

    def set_up_emulation(self):
        topology: NetworkTopology = NetworkTopology(subnet_to_nodes=self._subnet_to_nodes)
        net = Mininet(topo=topology, controller=None, switch=OVSBridge)
        net.start()

        # router are connected point to point
        router_links = [(nodes, sub_net) for sub_net, nodes in self._subnet_to_nodes.items()
                        if len(nodes) == 2
                        and nodes[0].node_type == NodeType.ROUTER
                        and nodes[1].node_type == NodeType.ROUTER]
        for flow, routes in self._flow_to_routes.items():
            flow_demand = self._flow_demands[flow]
            target_host_name: str = flow_demand.destination_node
            target_host: NodeDefinition = [n for nodes in self._subnet_to_nodes.values() for n in nodes
                                           if n.node_name == target_host_name][0]
            source_host_name: str = flow_demand.source_node
            source_host: NodeDefinition = [n for nodes in self._subnet_to_nodes.values() for n in nodes
                                           if n.node_name == source_host_name][0]
            print(f"flow {flow}: {flow_demand.source_node} - {flow_demand.destination_node} - {flow_demand.rate}")

            # flow table 0 is special, add 1 to avoid it
            mark = flow + 1
            immediate_router = self._get_connected_router(source_host)
            net[immediate_router.node_name].cmd(f"iptables -t mangle -A PREROUTING -s {source_host.address} -d {target_host.address} -j MARK --set-mark {mark}")
            net[immediate_router.node_name].cmd("sysctl net.ipv4.conf.all.rp_filter=0")
            for route in routes:
                source_router_name: str = route.split("_")[0]
                target_router_name: str = route.split("_")[1]
                source_router, target_router = self._get_node_definition_in_same_link(router_links,
                                                                                      source_router_name,
                                                                                      target_router_name)
                # old
                # routing_table_entry = f"ip route add {target_host.address} via {target_router.address}"
                routing_table_entry = f"ip route add {target_host.address} via {target_router.address} table {mark}"
                net[source_router_name].cmd(f"ip rule add fwmark {mark} table {mark} priority 1000")
                print(f"{source_router_name}: {routing_table_entry}")
                net[source_router_name].cmd(routing_table_entry)
                # old
                # routing_table_entry_t = f"ip route add {source_host.address} via {source_router.address}"
                routing_table_entry_t = f"ip route add {source_host.address} via {source_router.address} table {mark}"
                net[target_router_name].cmd(routing_table_entry_t)
                print(f"{target_router_name}: {routing_table_entry_t}")

        ###
        shortest_paths = self._find_shortest_paths()
        # HOST go through their only router anyway
        node_names = [n.node_name for v in self._subnet_to_nodes.values() for n in v if
                      n.node_type != NodeType.HOST]
        node_names = list(set(node_names))
        # set up routing tables
        for source_node, paths in shortest_paths.items():
            node_to_prev = paths[1]
            source_node_interfaces = [n for v in self._subnet_to_nodes.values() for n in v if
                                      n.node_name == source_node]
            for node_name in node_names:
                if node_name == source_node:
                    continue
                prev = node_to_prev[node_name]
                # prev and node are neighbours therefore they are both part of at least one subnet
                link = self.find_shortest_link_between(node_name, prev)

                for si in source_node_interfaces:
                    complete_subnet_address = si.complete_address()
                    routing_table_entry = f"ip route add {si.address} via {link.address}"
                    # print(f"{node_name}: {routing_table_entry}")
                    net[node_name].cmd(routing_table_entry)
        pass

        # for reasons beyond my understanding the hosts need to be told how to find other hosts explicitly even
        # if they have a default route.
        nodes = [n for v in self._subnet_to_nodes.values() for n in v]
        hosts: List[NodeDefinition] = [n for n in nodes if n.node_type == NodeType.HOST]
        for host in hosts:
            host_subnet = get_subnet(host.address, host.mask)
            subnet_nodes = self._subnet_to_nodes[host_subnet]
            router = [n for n in subnet_nodes if n.node_type == NodeType.ROUTER][0]
            for node in nodes:
                if host.node_name == node.node_name:
                    continue
                host2_subnet = get_subnet(node.address, node.mask)
                if host2_subnet == host_subnet:
                    continue
                net[host.node_name].cmd(f"ip route add {node.address} via {router.address}")

        CLI(net)
        net.stop()
        return

    def print_cplex(self):
        cplex_file: Path = self._get_cplex_file()
        with open(cplex_file, "r") as cplex:
            for line in cplex.readlines():
                print(line, end="")
        return

    def print_goodput(self):
        for flow, goodput in self._flow_to_goodput.items():
            print(f"The best goodput for flow demand #{flow} is {goodput} Mpbs")
        return


def main():
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="A tool to define the emulation a network "
                                                                          "configured to achieve the best overall "
                                                                          "goodput under a given set of flow demands.")
    parser.add_argument("definition", type=Path, help="the definition file of the network in YAML")
    parser.add_argument("--print", action="store_true",
                        help="print the optimal goodput for each flow and exit")
    parser.add_argument("--lp", action="store_true",
                        help="print the definition of the optimization problem in CPLEX LP format")
    args: argparse.Namespace = parser.parse_args()

    definition_file: Path = args.definition
    validate_definition_file(definition_file=definition_file)
    network_specification: dict = read_definition(definition_file=definition_file)
    network_definition: NetworkDefinition = NetworkDefinition(network_definition=network_specification)

    should_print_goodput: bool = args.print
    should_print_cplex: bool = args.lp
    if should_print_goodput:
        network_definition.print_goodput()
    if should_print_cplex:
        network_definition.print_cplex()

    network_definition.set_up_emulation()
    return


if __name__ == "__main__":
    # Author: Jacob Salvi
    # I thank the teaching assistant, Pasquale Polverino, for the help given during this assignment.
    # r1 ip route show table all
    main()
