import argparse
import math
import dataclasses
import ipaddress
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
    def __init__(self, subnet_to_nodes, subnet_to_cost: Dict[str, int]):
        self._subnet_to_nodes = subnet_to_nodes
        self._subnet_to_cost = subnet_to_cost
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
        self._load_routers(routers_def=routers)
        self._load_hosts(hosts_def=hosts)
        self._create_cplex_file()

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

    def _create_cplex_file(self):
        output_folder: Path = self._get_output_folder()
        output_folder.mkdir(exist_ok=True)
        cplex_definition_file: Path = self._get_cplex_file()
        if cplex_definition_file.is_file():
            cplex_definition_file.unlink()
        cplex_definition_file.touch()

        cplex_def: TextIO = open(cplex_definition_file, "w")
        cplex_def.writelines(["Maximize\n", "obj: rate_min\n", "\n"])

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

        # mutual exclusion of incoming flows into same node, for each flow, for each node
        subject_to.append(f"{indentation}\\ mutual exclusion of incoming flows into same node, for each flow, for each node")
        for idx, flow_demand in enumerate(self._flow_demands):
            for router_name in router_names:
                adjacent_routers = self._get_adjacent_routers(router_name=router_name)
                flow_balance = f"{indentation}in_{router_name}_{idx}: "
                tmp = []
                for adjacent_router in adjacent_routers:
                    tmp.append(f"fd{idx}_{adjacent_router}_{router_name}")
                flow_balance += " + ".join(tmp)
                flow_balance += " <= 1"
                subject_to.append(flow_balance)
            # Used to visually separate flow for easier inspection
            subject_to.append("")

        # mutual exclusion of outgoing out of same node, for each flow, for each node
        subject_to.append(f"{indentation}\\ mutual exclusion of outgoing out of same node, for each flow, for each node")
        for idx, flow_demand in enumerate(self._flow_demands):
            for router_name in router_names:
                adjacent_routers = self._get_adjacent_routers(router_name=router_name)
                flow_balance = f"{indentation}out_{router_name}_{idx}: "
                tmp = []
                for adjacent_router in adjacent_routers:
                    tmp.append(f"fd{idx}_{router_name}_{adjacent_router}")
                flow_balance += " + ".join(tmp)
                flow_balance += " <= 1"
                subject_to.append(flow_balance)
            # Used to visually separate flow for easier inspection
            subject_to.append("")

        # link capacities, for each link
        subject_to.append(f"{indentation}\\ link capacities, for each link")
        for router_name in router_names:
            adjacent_routers = self._get_adjacent_routers(router_name=router_name)
            for adjacent_router in adjacent_routers:
                tmp = []
                for idx, flow_demand in enumerate(self._flow_demands):
                    pass


        # Binary section
        binaries: List[str] = ["Binary"]

        subject_to = [f"{line}\n" for line in subject_to]
        binaries = [f"{line}\n" for line in binaries]
        cplex_def.writelines(subject_to)
        cplex_def.writelines(binaries)
        cplex_def.write("End")
        cplex_def.close()
        return

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

    def set_up_emulation(self):
        shortest_paths = self._find_shortest_paths()
        topology: NetworkTopology = NetworkTopology(subnet_to_nodes=self._subnet_to_nodes,
                                                    subnet_to_cost=self._subnet_to_cost)

        # r2 ip route add 192.168.0.4/30 via 192.168.0.1
        # r3 ip route add 192.168.0.0/30 via 192.168.0.5

        net = Mininet(topo=topology, controller=None, switch=OVSBridge)
        net.start()

        # HOST go through their only router anyway
        node_names = [n.node_name for v in self._subnet_to_nodes.values() for n in v if n.node_type != NodeType.HOST]
        node_names = list(set(node_names))
        # set up routing tables
        for source_node, paths in shortest_paths.items():
            # if source_node.startswith("h"):
            #     continue
            node_to_dist = paths[0]
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
                    print(f"{node_name}: {routing_table_entry}")
                    net[node_name].cmd(routing_table_entry)
                pass
            #  router1.cmd('ip route add 10.0.2.0/24 via 10.1.2.2')
            pass
        pass

        # for reasons beyond my understanding the hosts need to be told how to find other hosts explicitly even
        # if they have a default route.
        nodes = [n for v in self._subnet_to_nodes.values() for n in v]
        hosts: List[NodeDefinition] = [n for n in nodes if n.node_type == NodeType.HOST]
        for host in hosts:
            host_subnet = get_subnet(host.address, host.mask)
            subnet_nodes = self._subnet_to_nodes[host_subnet]
            router = [n for n in subnet_nodes if n.node_type == NodeType.ROUTER][0]
            for host2 in hosts:
                if host.node_name == host2.node_name:
                    continue
                host2_subnet = get_subnet(host2.address, host2.mask)
                if host2_subnet == host_subnet:
                    continue
                net[host.node_name].cmd(f"ip route add {host2.address} via {router.address}")

        # r2 = net["r2"]
        # r3 = net["r3"]
        # r2.cmd("ip route add 10.0.3.0/24 via 192.168.1.3")
        # r3.cmd("ip route add 10.0.2.0/24 via 192.168.1.2")

        # r2.cmd("ip route add 192.168.0.4/30 via 192.168.0.1")
        # r3.cmd("ip route add 192.168.0.0/30 via 192.168.0.5")

        CLI(net)
        net.stop()
        return

    def print_cplex(self):
        cplex_file: Path = self._get_cplex_file()
        with open(cplex_file, "") as cplex:
            for line in cplex.readlines():
                print(line)
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
        # network_definition.output_graph()
        pass
    if should_print_cplex:
        network_definition.print_cplex()

    # network_definition.set_up_emulation()
    return


if __name__ == "__main__":
    # Author: Jacob Salvi
    # I thank the teaching assistant, Pasquale Polverino, for the help given during this assignment.
    main()