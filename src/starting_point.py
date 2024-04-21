#!/usr/bin/env python3

import argparse
import itertools
import math
import re
import sys
import yaml

from mininet.cli import CLI
from mininet.link import TCLink
from mininet.log import setLogLevel
from mininet.net import Mininet
from mininet.node import Node
from mininet.topo import Topo


DEFAULT_COST = 1


def network_id(ip, mask):
    ip_list, mask_list = ip.split("."), mask.split(".")
    cidr = sum([bin(int(i)).count("1") for i in mask_list])
    for i in range(len(ip_list)):
        ip_list[i] = str(int(ip_list[i]) & int(mask_list[i]))
    return ".".join(ip_list) + "/" + str(cidr)


def prefix_len(mask):
    mask_list = mask.split(".")
    return sum([bin(int(i)).count("1") for i in mask_list])


def compute_broadcast_domains(nodes):
    domains = {}

    for hostname, ifaces in nodes.items():

        for ifname, address in ifaces.items():
            net_addr = network_id(address["address"], address["mask"])

            if net_addr not in domains:
                domains[net_addr] = []

            domains[net_addr].append(
                (
                    hostname,
                    ifname,
                    f"{address['address']}/{prefix_len(address['mask'])}",
                )
            )

    return domains


def compute_link_costs(domains, routers):
    global DEFAULT_COST

    costs = {}

    for domain, ifaces in domains.items():
        cost = DEFAULT_COST
        for host, ifname, _ in ifaces:
            if "cost" in routers[host][ifname]:
                cost = routers[host][ifname]["cost"]
                break

        costs[domain] = cost

    return costs


class Router(Node):
    def config(self, **params):
        super(Router, self).config(**params)
        self.cmd("sysctl net.ipv4.ip_forward=1")

    def terminate(self):
        self.cmd("sysctl net.ipv4.ip_forward=0")
        super(Router, self).terminate()


class Topology(Topo):

    def build(self, **params):
        network = params["definition"]
        domains = compute_broadcast_domains({**network["hosts"], **network["routers"]})

        default_host_routes = {}
        for domain, ifaces in domains.items():
            for hostname, _, addr in ifaces:
                if hostname in network["routers"]:
                    default_host_routes[domain] = addr.split("/")[0]

        for router in network["routers"]:
            self.addNode(router, cls=Router, ip=None)

        for host, ifaces in network["hosts"].items():
            default_route = None
            for _, addr in ifaces.items():
                domain = network_id(addr["address"], addr["mask"])
                if domain in default_host_routes:
                    default_route = f"via {default_host_routes[domain]}"
                    break

            self.addHost(host, ip=None, defaultRoute=default_route)

        switches = 0
        for domain, ifaces in domains.items():
            if len(ifaces) == 2:
                node1, iface1, addr1 = ifaces[0]
                node2, iface2, addr2 = ifaces[1]
                self.addLink(
                    node1,
                    node2,
                    intfName1=f"{node1}-{iface1}",
                    intfName2=f"{node2}-{iface2}",
                    params1={"ip": addr1},
                    params2={"ip": addr2},
                )
            else:
                switch = self.addSwitch(f"s{switches}")

                switches += 1
                ifnumber = 0

                for host, ifname, addr in ifaces:
                    self.addLink(
                        switch,
                        host,
                        intfName1=f"{switch}-eth{ifnumber}",
                        intfName2=f"{host}-{ifname}",
                        params2={"ip": addr},
                    )
                    ifnumber += 1


def main():
    parser = argparse.ArgumentParser(
        description="A tool to define the emulation of a network."
    )

    parser.add_argument(
        "definition",
        type=argparse.FileType("r"),
        help="the definition file of the network in YAML",
    )

    args = parser.parse_args()

    try:
        network_definition = yaml.safe_load(args.definition)
    except:
        print("The definition file is not a valid YAML file", file=sys.stderr)
        return 1

    topology = Topology(definition=network_definition)
    net = Mininet(topo=topology, link=TCLink)

    net.start()
    CLI(net)
    net.stop()


if __name__ == "__main__":
    setLogLevel("info")
    sys.exit(main())
