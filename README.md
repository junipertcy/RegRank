# rSpringRank

[![license](https://img.shields.io/badge/license-LGPL-green.svg?style=flat)](https://github.com/junipertcy/rSpringRank/blob/main/LICENSE)


**rSpringRank** implements a collection of regularized, convex models (+solvers) that allow the inference of hierarchical structure in a directed network, which exists due to dominance, social status, or prestige. Specifically, this work leverages the time-varying structure and/or the node metadata present in the data set.

This is the software repository behind the paper:
* Tzu-Chi Yen and Stephen Becker, *Regularized methods for efficient ranking in networks*, in preparation.


* For full documentation, please visit [this site](https://).
* For general Q&A, ideas, or other things, please visit [Discussions](https://).
* For software-related bugs, issues, or suggestions, please use [Issues](https://).



Installation
------------
**rSpringRank** is available on PyPI:
```
pip install rSpringRank
```

The dependencies needed to use graph-tool-based input are not installed by default. You will need to install:
* Graph-tool: https://graph-tool.skewed.de/


First steps
-----------
In your Python console, `rSpringRank` is invoked using:

```python
>>> from reg_sr import rSpringRank, PhDExchange
>>> pde = PhDExchange()
>>> g = pde.get_data(goi="c18basic")
>>> rsp = rSpringRank(method="annotated")
>>> result = rsp.fit(g, alpha=1, lambd=1, printEvery=0)  # actual computation; takes ~5 seconds
>>> pde.compute_basic_stats(rsp.sslc, primal_s=result["primal"])
>>> pde.print_sorted_mean(5, precision=3)  # output the higher mean SpringRank categories
Group: 15; Mean: 0.171
Group: na; Mean: -0.043
Group: 22; Mean: -0.043
Group: 18; Mean: -0.043
Group: 21; Mean: -0.043
```

As you noticed, most groups have the same mean SpringRank. This is the regularizer in effect. To plot the distribution, do this.

```
>>> pde.plot_hist(bin_count=20, legend=True)
```

![A histogram of ranks stratified by C18 category](etc/example_c18.png)



### Development
The library uses pytest to ensure correctness. The test suite depends on [mosek](https://www.mosek.com/) and [gurobi](https://www.gurobi.com/).

```python
pytest
```


Related links
-------------
TODO.

Acknowledgement
---------------
TODO.


License
-------
**rSpringRank** is open-source and licensed under the [GNU Lesser General Public License v3.0](https://www.gnu.org/licenses/lgpl-3.0.en.html).
