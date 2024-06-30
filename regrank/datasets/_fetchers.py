from ._registry import registry, registry_urls

# import networkx as nx
try:
    import graph_tool.all as gt
except ModuleNotFoundError:
    print("We need graph-tool to load the datasets. Please install graph-tool.")

import linecache
import tempfile
from pathlib import Path

import numpy as np
import zstandard as zstd

try:
    import pooch
except ImportError:
    pooch = None
    data_fetcher = None
else:
    data_fetcher = pooch.create(
        # Use the default cache folder for the operating system
        # Pooch uses appdirs (https://github.com/ActiveState/appdirs) to
        # select an appropriate directory for the cache on each platform.
        path=pooch.os_cache("rSpringRank-data"),
        # The remote data is on Github
        # base_url is a required param, even though we override this
        # using individual urls in the registry.
        base_url="https://github.com/junipertcy/",
        registry=registry,
        urls=registry_urls,
    )


def fetch_data(dataset_name, data_fetcher=data_fetcher):
    if data_fetcher is None:
        raise ImportError(
            "Missing optional dependency 'pooch' required "
            "for scipy.datasets module. Please use pip or "
            "conda to install 'pooch'."
        )
    # The "fetch" method returns the full path to the downloaded data file.
    return data_fetcher.fetch(dataset_name)


def small_graph():
    # This is a small graph for debugging
    # This should generate a list of SpringRank (alpha=1) of
    # [-0.57142857, -0.14285714, 0.14285714, 0.57142857]
    # See https://github.com/LarremoreLab/SpringRank/pull/3
    g = gt.Graph()
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    return g


def random_graph():
    def _deg_sample():
        if np.random.rand() > 0.5:
            return np.random.poisson(4), np.random.poisson(4)
        else:
            return np.random.poisson(8), np.random.poisson(8)

    N = np.random.randint(10, 100)
    g = gt.random_graph(N, _deg_sample, directed=True)

    return g


def us_air_traffic():
    # The file will be downloaded automatically the first time this is run,
    # returning the path to the downloaded file. Afterwards, Pooch finds
    # it in the local cache and doesn't repeat the download.
    fname = fetch_data("us_air_traffic.gt.zst")
    # Now we just need to load it with our standard Python tools.
    fname = Path(fname)
    dctx = zstd.ZstdDecompressor()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        gt_fname = temp_dir_path / fname.with_suffix("").name
        with open(fname, "rb") as ifh, open(gt_fname, "wb") as ofh:
            dctx.copy_stream(ifh, ofh)
        graph = gt.load_graph(gt_fname.as_posix())
        return graph


def at_migrations():
    fname = fetch_data("at_migrations.gt.zst")
    fname = Path(fname)
    dctx = zstd.ZstdDecompressor()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        gt_fname = temp_dir_path / fname.with_suffix("").name
        with open(fname, "rb") as ifh, open(gt_fname, "wb") as ofh:
            dctx.copy_stream(ifh, ofh)
        graph = gt.load_graph(gt_fname.as_posix())
        return graph


def PhD_exchange():
    fname = fetch_data("PhD_exchange.txt")
    fname_school_names = fetch_data("school_names.txt")
    delimiter = " "

    g = gt.Graph()
    vname = g.new_vp("string")
    vindex = g.new_vp("int")
    eweight = g.new_ep("double")
    etime = g.new_ep("int")

    name2id = dict()
    time2id = dict()
    nameid = 0
    timeid = 0

    with open(fname) as f:
        for line in f:
            ijwt = line.replace("\n", "").split(delimiter)[:4]

            try:
                name2id[ijwt[0]]
            except KeyError:
                name2id[ijwt[0]] = nameid
                nameid += 1

            try:
                name2id[ijwt[1]]
            except KeyError:
                name2id[ijwt[1]] = nameid
                nameid += 1

            try:
                time2id[ijwt[3]]
            except KeyError:
                time2id[ijwt[3]] = timeid
                timeid += 1

            g.add_edge_list(
                [
                    (name2id[ijwt[1]], name2id[ijwt[0]], ijwt[2], time2id[ijwt[3]])
                ],  # note the source / target order
                eprops=[eweight, etime],
            )
    g.edge_properties["eweight"] = eweight
    g.edge_properties["etime"] = etime
    id2name = {v: k for k, v in name2id.items()}

    def school_name(n):
        return linecache.getline(fname_school_names, n).replace("\n", "")[:-1]

    # print(school_name(165))  # >> University of Michigan
    for vertex in g.vertices():
        vname[vertex] = school_name(int(id2name[vertex]))
        # print(vname[vertex], vertex, id2name[vertex])
        vindex[vertex] = vertex.__int__()

    g.vertex_properties["vname"] = vname
    g.vertex_properties["vindex"] = vindex
    # print(name2id)

    return g


def parakeet(group="G1"):
    """_summary_

    Args:
        group (str, optional): Group of parakeets, either "G1" or "G2". Defaults to "G1".

    Returns:
        _type_: _description_
    """
    fname = fetch_data("aggXquarter.txt")
    g = gt.Graph()
    name2id = dict()
    id_counter = 0
    qtrs = []
    with open(fname) as f:
        lines = f.readlines()
        for line in lines:
            # src: actor; tar: target
            grp, qtr, src, tar, wins = line.split()
            if grp != group:
                continue
            try:
                name2id[src]
            except KeyError:
                name2id[src] = id_counter
                id_counter += 1

            try:
                name2id[tar]
            except KeyError:
                name2id[tar] = id_counter
                id_counter += 1

            for _ in range(int(wins)):
                g.add_edge(
                    name2id[tar], name2id[src]
                )  # actor wins, so we reverse the edge
                qtrs.append(str(qtr))

    id2name = {v: k for k, v in name2id.items()}

    g.vp["name"] = g.new_vertex_property("string")
    for node in g.vertices():
        g.vp["name"][node] = id2name[node]

    g.ep["quarter"] = g.new_edge_property("string")
    for eid, _ in enumerate(g.edges()):
        g.ep["quarter"][_] = qtrs[eid]

    return g
