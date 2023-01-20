import graph_tool.all as gt


def render_ijwt(path="./etc/prestige_reinforcement/data/PhD Exchange Network Data/PhD_exchange.txt", delimiter=" "):
    g = gt.Graph()
    eweight = g.new_ep("double")
    etime = g.new_ep("int")

    name2id = dict()
    time2id = dict()
    nameid = 0
    timeid = 0

    with open(path, "r") as f:
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

            g.add_edge_list([
                (name2id[ijwt[0]], name2id[ijwt[1]], ijwt[2], time2id[ijwt[3]])
            ], eprops=[eweight, etime])
    g.edge_properties["eweight"] = eweight
    g.edge_properties["etime"] = etime
    return g


def filter_by_time(g, time):
    mask_e = g.new_edge_property("bool")
    for edge in g.edges():
        if g.ep["etime"][edge] == time:
            mask_e[edge] = 1
        else:
            mask_e[edge] = 0
    return mask_e
