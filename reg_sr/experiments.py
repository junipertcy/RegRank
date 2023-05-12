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
        self.sslc = None
        # Basic stats
        self.num_classes = 0
        self.num_dual_vars = 0
        self.num_primal_vars = 0
        self.data_annot = None
        self.basic_stats = defaultdict(int)
        pass

    def get_data(self, annot="sector"):
        """
        annot: which stratum (metadata of the nodes) that you are looking for?

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

        self.g.vp["class"] = self.g.new_vertex_property("string")

        # np.random.seed(1)  # debug use
        for node in self.g.vertices():
            pi_id = client["pnas2015188118"]["phd-exchange-nodes"].find_one(
                {"unitid": str(int(node) + 1)}
            )["peer-institutions-nodes_id"]
            if pi_id == -1:
                self.g.vp["class"][node] = "na"

                # debug use
                # if np.random.random() <= 0.5:
                #     self.g.vp["class"][node] = "na1"
                # else:
                #     self.g.vp["class"][node] = "na2"
                # self.g.vp["class"][node] = "2"

            else:
                data = client["peer-institutions"]["nodes"].find_one({"_id": pi_id})
                self.g.vp["class"][node] = data[annot]

        # print(to_remove_in := np.where(self.g.get_in_degrees(list(self.g.vertices())) == 0))
        # print(to_remove_out := np.where(self.g.get_out_degrees(list(self.g.vertices())) == 0))
        to_remove_total = np.where(
            self.g.get_total_degrees(list(self.g.vertices())) == 0
        )

        for node in to_remove_total[0]:
            self.g.remove_vertex(node, fast=1)

        # construct basic stats
        self.num_classes = len(set(np.array(list(self.g.vp["class"]))))
        self.num_dual_vars = comb(self.num_classes, 2)
        self.num_primal_vars = self.g.num_vertices()

        return self.g

    def get_node_metadata(self):
        return np.array(list(self.g.vp["class"]))

    def draw(self):
        # state = gt.minimize_nested_blockmodel_dl(self.g)
        # gt.draw_hierarchy(state)
        gt.graph_draw(self.g)

    def plot_hist(self, bin_count=20, legend=False):
        hstack = []
        for key in self.data_annot.keys():
            hstack.append(self.data_annot[key])

        bins = np.histogram(np.concatenate(hstack), bins=bin_count)[1]

        for key in self.data_annot.keys():
            plt.hist(self.data_annot[key], bins, label=key, alpha=0.5)
        if legend:
            plt.legend()

    def _compute_collection_by_annot(self, dual_v=None, primal_s=None):
        if dual_v is not None and primal_s is not None:
            raise AttributeError("Only use either dual_v or primal_s.")
        elif dual_v is None and primal_s is None:
            raise AttributeError("You need some input data.")
        elif dual_v is not None:
            if self.sslc is None:
                self.sslc = self._init_sslc()
            # We take firstOrderMethods.py output directly
            dual_v = np.array(dual_v).reshape(-1, 1)
            output = self.sslc.dual2primal(dual_v)
        else:
            output = primal_s
        collection_by_annot = defaultdict(list)
        for idx, _c in enumerate(self.get_node_metadata()):
            collection_by_annot[_c].append(output[idx])
        self.data_annot = collection_by_annot
        return collection_by_annot

    def _init_sslc(self):
        sslc = sum_squared_loss_conj()
        sslc.setup(self.g, alpha=1)
        return sslc

    def compute_Ls(self, primal_s):
        if self.sslc is None:
            self.sslc = self._init_sslc()

        # TCY comment: this is almost the same as the firstOrder.py result
        # this data should be very sparse (mostly zero)
        return self.sslc.ell @ primal_s.value

    def dual2primal(self, dual_v):
        if self.sslc is None:
            self.sslc = self._init_sslc()
        if type(dual_v) is np.matrix:
            return self.sslc.dual2primal(np.array(dual_v).reshape(-1, 1))
        return self.sslc.dual2primal(dual_v.value)

    def compute_basic_stats(self, dual_v=None, primal_s=None):
        self.basic_stats["deviation_dict"] = dict()
        if self.data_annot is None:
            self._compute_collection_by_annot(dual_v=dual_v, primal_s=primal_s)

        keys_comb = combinations(self.data_annot.keys(), 2)
        for key_pair in keys_comb:
            key_0, key_1 = key_pair[0], key_pair[1]
            diff = np.mean(self.data_annot[key_0]) - np.mean(self.data_annot[key_1])
            if np.abs(diff) >= 1e-5:
                self.basic_stats["nonzero"] += 1
            if key_0 != key_1:
                self.basic_stats["total"] += 1
            self.basic_stats["deviation_dict"][key_pair] = diff

        self.basic_stats["mean_dict"] = dict()
        for col_key in self.data_annot.keys():
            self.basic_stats["mean_dict"][col_key] = np.mean(self.data_annot[col_key])

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
    # TODO!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def __init__(self):
        self.g = gt.Graph()
        pass

    def get_data(self, annot="sector"):
        """
        annot: which stratum (metadata of the nodes) that you are looking for?

        """
        d = client["peer-institutions"]["edges"].find({})

        for _d in d:
            src = int(_d["source"]) - 1
            tar = int(_d["target"]) - 1
            if tar != src:
                self.g.add_edge(src, tar)

        #     if np.random.random() < 0.1:
        #         self.g.add_edge(np.random.randint(231), np.random.randint(231))

        # pi_id = client["pnas2015188118"]["phd-exchange-nodes"].find_one({"unitid": str(src + 1)})["peer-institutions-nodes_id"]
        # print(pi_id)

        self.g.vp["class"] = self.g.new_vertex_property("string")

        # np.random.seed(1)  # debug use
        for node in self.g.vertices():
            pi_id = client["pnas2015188118"]["phd-exchange-nodes"].find_one(
                {"unitid": str(int(node) + 1)}
            )["peer-institutions-nodes_id"]
            if pi_id == -1:
                self.g.vp["class"][node] = "na"

                # debug use
                # if np.random.random() <= 0.5:
                #     self.g.vp["class"][node] = "na1"
                # else:
                #     self.g.vp["class"][node] = "na2"
                # self.g.vp["class"][node] = "2"

            else:
                data = client["peer-institutions"]["nodes"].find_one({"_id": pi_id})
                self.g.vp["class"][node] = data[annot]

        # print(to_remove_in := np.where(self.g.get_in_degrees(list(self.g.vertices())) == 0))
        # print(to_remove_out := np.where(self.g.get_out_degrees(list(self.g.vertices())) == 0))
        print(
            to_remove_total := np.where(
                self.g.get_total_degrees(list(self.g.vertices())) == 0
            )
        )

        for node in to_remove_total[0]:
            self.g.remove_vertex(node, fast=1)

        return self.g

    def get_node_metadata(self):
        return np.array(list(self.g.vp["class"]))

    def draw(self):
        # state = gt.minimize_nested_blockmodel_dl(self.g)
        # gt.draw_hierarchy(state)
        gt.graph_draw(self.g)

    def plot_hist(self, dual_v=None, primal_s=None, bin_count=20, legend=False):
        if dual_v is not None and primal_s is not None:
            raise AttributeError("Only use either dual_v or primal_s.")
        elif dual_v is None and primal_s is None:
            raise AttributeError("You need some input data.")
        elif dual_v:
            output = sslc.dual2primal(xNew)
        else:
            output = primal_s

        collection_by_annot = defaultdict(list)
        for idx, _c in enumerate(self.get_node_metadata()):
            collection_by_annot[_c].append(output[idx])

        hstack = []
        for key in collection_by_annot.keys():
            hstack.append(collection_by_annot[key])

        bins = np.histogram(np.concatenate(hstack), bins=bin_count)[1]

        for key in collection_by_annot.keys():
            plt.hist(collection_by_annot[key], bins, label=key, alpha=0.5)
        if legend:
            plt.legend()
        return collection_by_annot
