#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Regularized-SpringRank -- regularized methods for efficient ranking in networks
#
# Copyright (C) 2023 Tzu-Chi Yen <tzuchi.yen@colorado.edu>
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

from bson import ObjectId
from pymongo import MongoClient

username = "tzuchi"
password = "FI4opd12cjazqPL"
port = "127.0.0.1:27017"

client = MongoClient("mongodb://%s:%s@%s" % (username, password, port))

import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import combinations
import graph_tool.all as gt
import numpy as np
from math import comb
from reg_sr.losses import *


# TODO, see
# https://graph-tool.skewed.de/static/doc/quickstart.html#sec-graph-filtering
class Experiment:
    def __init__(self):
        pass

    def get_data(self):
        pass

    def draw(self):
        pass


class SmallGraph(Experiment):
    # This is a small graph for debugging
    # This should generate a list of SpringRank (alpha=1) of
    # [-0.57142857, -0.14285714, 0.14285714, 0.57142857]
    # See https://github.com/LarremoreLab/SpringRank/pull/3
    def __init__(self):
        self.g = gt.Graph()
        self.g.add_edge(0, 1)
        self.g.add_edge(1, 2)
        self.g.add_edge(2, 3)
        pass

    def get_data(self):
        return self.g


class RandomGraph(Experiment):
    def __init__(self):
        N = np.random.randint(10, 100)
        self.g = gt.random_graph(N, self._deg_sample, directed=True)

    def get_data(self):
        return self.g

    @staticmethod
    def _deg_sample():
        if np.random.rand() > 0.5:
            return np.random.poisson(4), np.random.poisson(4)
        else:
            return np.random.poisson(8), np.random.poisson(8)


class PhDExchange(Experiment):
    def __init__(self):
        self.g = gt.Graph()
        # Basic stats
        self.num_classes = 0
        self.num_dual_vars = 0
        self.num_primal_vars = 0
        self.data_goi = None
        self.basic_stats = defaultdict(int)
        pass

    def get_data(self, goi="sector"):
        """
        goi: which stratum (metadata of the nodes) that you are looking for?

        """
        d = client["pnas2015188118"]["phd-exchange-edges"].find({})

        for _d in d:
            src = int(_d["source"]) - 1
            tar = int(_d["target"]) - 1
            if tar != src:
                self.g.add_edge(src, tar)

        #     if np.random.random() < 0.1:
        #         self.g.add_edge(np.random.randint(231), np.random.randint(231))

        # pi_id = client["pnas2015188118"]["phd-exchange-nodes"].find_one({"unitid": str(src + 1)})["peer-institutions-nodes_id"]
        # print(pi_id)

        self.g.vp["goi"] = self.g.new_vertex_property("string")

        # np.random.seed(1)  # debug use
        for node in self.g.vertices():
            pi_id = client["pnas2015188118"]["phd-exchange-nodes"].find_one(
                {"unitid": str(int(node) + 1)}
            )["peer-institutions-nodes_id"]
            if pi_id == -1:
                self.g.vp["goi"][node] = "na"

                # debug use
                # if np.random.random() <= 0.5:
                #     self.g.vp["goi"][node] = "na1"
                # else:
                #     self.g.vp["goi"][node] = "na2"
                # self.g.vp["goi"][node] = "2"

            else:
                data = client["peer-institutions"]["nodes"].find_one({"_id": pi_id})
                self.g.vp["goi"][node] = data[goi]

        # print(to_remove_in := np.where(self.g.get_in_degrees(list(self.g.vertices())) == 0))
        # print(to_remove_out := np.where(self.g.get_out_degrees(list(self.g.vertices())) == 0))
        to_remove_total = np.where(
            self.g.get_total_degrees(list(self.g.vertices())) == 0
        )

        for node in to_remove_total[0]:
            self.g.remove_vertex(node, fast=1)

        # construct basic stats
        self.num_classes = len(set(np.array(list(self.g.vp["goi"]))))
        self.num_dual_vars = comb(self.num_classes, 2)
        self.num_primal_vars = self.g.num_vertices()

        return self.g

    def get_node_metadata(self):
        return np.array(list(self.g.vp["goi"]))

    def draw(self):
        # state = gt.minimize_nested_blockmodel_dl(self.g)
        # gt.draw_hierarchy(state)
        gt.graph_draw(self.g)

    def plot_hist(self, bin_count=20, legend=False):
        hstack = []
        for key in self.data_goi.keys():
            hstack.append(self.data_goi[key])

        bins = np.histogram(np.concatenate(hstack), bins=bin_count)[1]

        for key in self.data_goi.keys():
            plt.hist(self.data_goi[key], bins, label=key, alpha=0.5)
        if legend:
            plt.legend()

    def _compute_collection_by_goi(self, sslc, dual_v=None, primal_s=None):
        if dual_v is not None and primal_s is not None:
            raise AttributeError("Only use either dual_v or primal_s.")
        elif dual_v is None and primal_s is None:
            raise AttributeError("You need some input data.")
        elif dual_v is not None:
            # We take firstOrderMethods.py output directly
            dual_v = np.array(dual_v).reshape(-1, 1)
            output = sslc.dual2primal(dual_v)
        else:
            output = primal_s
        collection_by_goi = defaultdict(list)
        for idx, _c in enumerate(self.get_node_metadata()):
            collection_by_goi[_c].append(output[idx])
        self.data_goi = collection_by_goi
        return collection_by_goi

    def compute_basic_stats(self, sslc, dual_v=None, primal_s=None):
        self.basic_stats["deviation_dict"] = dict()
        if self.data_goi is None:
            self._compute_collection_by_goi(sslc, dual_v=dual_v, primal_s=primal_s)

        keys_comb = combinations(self.data_goi.keys(), 2)
        for key_pair in keys_comb:
            key_0, key_1 = key_pair[0], key_pair[1]
            diff = np.mean(self.data_goi[key_0]) - np.mean(self.data_goi[key_1])
            if np.abs(diff) >= 1e-5:
                self.basic_stats["nonzero"] += 1
            if key_0 != key_1:
                self.basic_stats["total"] += 1
            self.basic_stats["deviation_dict"][key_pair] = diff

        self.basic_stats["mean_dict"] = dict()
        for col_key in self.data_goi.keys():
            self.basic_stats["mean_dict"][col_key] = np.mean(self.data_goi[col_key])

        self.basic_stats["sparsity_perc"] = (
            self.basic_stats["nonzero"] / self.basic_stats["total"]
        )

    def print_sorted_diff(self, num=5):
        return sorted(
            self.basic_stats["deviation_dict"].items(),
            key=lambda item: item[1],
            reverse=False,
        )[:num]

    def print_sorted_mean(self, num=5):
        return sorted(
            self.basic_stats["mean_dict"].items(),
            key=lambda item: item[1],
            reverse=True,
        )[:num]


class PeerInstitution(Experiment):
    def __init__(self):
        self.g = gt.Graph()
        # Basic stats
        self.num_classes = 0
        self.num_dual_vars = 0
        self.num_primal_vars = 0
        self.data_goi = None
        self.basic_stats = defaultdict(int)
        pass

    def get_data(self, goi="sector"):
        """
        goi: which stratum (metadata of the nodes) that you are looking at?

        """
        d = client["peer-institutions"]["edges"].find(
            {}, {"_id": 0, "unitid": 1, "peer_unitid": 1}
        )
        unitid2nodeid = dict()
        counter = 0
        unitid_set = set()
        peer_unitid_set = set()
        for _d in d:
            _unitid = _d["unitid"]
            
            if not _unitid in unitid_set:
                result = client["peer-institutions"]["nodes"].find_one(
                    {"unitid": str(_unitid)}
                )
                if result is None:
                    continue
                else:
                    unitid_set.add(_unitid)

            src = int(_unitid)
            if src not in unitid2nodeid:
                unitid2nodeid[src] = counter
                counter += 1

            _peer_unitid = _d["peer_unitid"]

            if not _peer_unitid in peer_unitid_set:
                result = client["peer-institutions"]["nodes"].find_one(
                    {"unitid": str(_peer_unitid)}
                )
                if result is None:
                    continue
                else:
                    peer_unitid_set.add(_peer_unitid)

            tar = int(_peer_unitid)
            if tar not in unitid2nodeid:
                unitid2nodeid[tar] = counter
                counter += 1

            if unitid2nodeid[tar] != unitid2nodeid[src]:
                self.g.add_edge(unitid2nodeid[src], unitid2nodeid[tar])
        inv_unitid2nodeid = {v: k for k, v in unitid2nodeid.items()}
        #     if np.random.random() < 0.1:
        #         self.g.add_edge(np.random.randint(231), np.random.randint(231))

        # pi_id = client["pnas2015188118"]["phd-exchange-nodes"].find_one({"unitid": str(src + 1)})["peer-institutions-nodes_id"]
        # print(pi_id)

        self.g.vp["goi"] = self.g.new_vertex_property("string")
        self.g.vp["instnm"] = self.g.new_vertex_property("string")

        # np.random.seed(1)  # debug use
        for node in self.g.vertices():
            pi_id = client["peer-institutions"]["nodes"].find_one(
                {"unitid": str(inv_unitid2nodeid[node])}
            )
            if pi_id is None:
                print({"unitid": str(inv_unitid2nodeid[node])})
            self.g.vp["goi"][node] = pi_id[goi]
            self.g.vp["instnm"][node] = pi_id["instnm"]


        return self.g

    def compute_basic_stats(self, sslc, dual_v=None, primal_s=None):
        self.basic_stats["deviation_dict"] = dict()
        if self.data_goi is None:
            self._compute_collection_by_goi(sslc, dual_v=dual_v, primal_s=primal_s)

        keys_comb = combinations(self.data_goi.keys(), 2)
        for key_pair in keys_comb:
            key_0, key_1 = key_pair[0], key_pair[1]
            diff = np.mean(self.data_goi[key_0]) - np.mean(self.data_goi[key_1])
            if np.abs(diff) >= 1e-5:
                self.basic_stats["nonzero"] += 1
            if key_0 != key_1:
                self.basic_stats["total"] += 1
            self.basic_stats["deviation_dict"][key_pair] = diff

        self.basic_stats["mean_dict"] = dict()
        for col_key in self.data_goi.keys():
            self.basic_stats["mean_dict"][col_key] = np.mean(self.data_goi[col_key])

        self.basic_stats["sparsity_perc"] = (
            self.basic_stats["nonzero"] / self.basic_stats["total"]
        )

    def get_node_metadata(self):
        return np.array(list(self.g.vp["goi"]))

    def draw(self):
        # state = gt.minimize_nested_blockmodel_dl(self.g)
        # gt.draw_hierarchy(state)
        gt.graph_draw(self.g)

    def plot_hist(self, bin_count=20, legend=False):
        hstack = []
        for key in self.data_goi.keys():
            hstack.append(self.data_goi[key])

        bins = np.histogram(np.concatenate(hstack), bins=bin_count)[1]

        for key in self.data_goi.keys():
            plt.hist(self.data_goi[key], bins, label=key, alpha=0.5)
        if legend:
            plt.legend()

    def _compute_collection_by_goi(self, sslc, dual_v=None, primal_s=None):
        if dual_v is not None and primal_s is not None:
            raise AttributeError("Only use either dual_v or primal_s.")
        elif dual_v is None and primal_s is None:
            raise AttributeError("You need some input data.")
        elif dual_v is not None:
            # We take firstOrderMethods.py output directly
            dual_v = np.array(dual_v).reshape(-1, 1)
            output = sslc.dual2primal(dual_v)
        else:
            output = primal_s
        collection_by_goi = defaultdict(list)
        for idx, _c in enumerate(self.get_node_metadata()):
            collection_by_goi[_c].append(output[idx])
        self.data_goi = collection_by_goi
        return collection_by_goi

    def print_sorted_diff(self, num=5):
        return sorted(
            self.basic_stats["deviation_dict"].items(),
            key=lambda item: item[1],
            reverse=False,
        )[:num]

    def print_sorted_mean(self, num=5):
        return sorted(
            self.basic_stats["mean_dict"].items(),
            key=lambda item: item[1],
            reverse=True,
        )[:num]
