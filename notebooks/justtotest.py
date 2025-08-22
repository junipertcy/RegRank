import sys

import graph_tool.all as gt
import rSpringRank as sr

sys.path.append("..")
# g = gt.collection.ns["wiki_rfa"] # 11,381; 198,275
# # g = gt.collection.ns["python_dependency"]  # 58,743; 108,399
# # g = gt.collection.ns["berkstan_web"]  # 685,231; 7,600,595
# print(g.list_properties())
# rsp = sr.optimize.rSpringRank(method="vanilla")
# print(rsp.fit(g)["primal"])
# # sp = sr.optimize.SpringRank(alpha=1)
# # print(sp.fit(g)["rank"])
g = sr.datasets.us_air_traffic()
_g = gt.GraphView(g, efilt=lambda e: g.ep["year"][e] == 2000)
rsr = sr.optimize.rSpringRank(method="annotated")
result = rsr.fit(_g, alpha=1.0, lambd=10, goi="state_abr")
