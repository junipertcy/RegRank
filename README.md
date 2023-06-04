# Regularized-SpringRank

[![license](https://img.shields.io/badge/license-LGPL-green.svg?style=flat)](https://github.com/junipertcy/Regularized-SpringRank/blob/main/COPYING)


**Regularized_SpringRank** implements a collection of regularized, convex models (+solvers) that allow the inference of hierarchical structure in a directed network, which exists due to dominance, social status, or prestige. Specifically, this work leverages the time-varying structure and/or the node metadata present in the data set.

This is the software repository behind the paper:
* Tzu-Chi Yen and Stephen Becker, *Regularized methods for efficient ranking in networks*, in preparation.


* For full documentation, please visit [this site](https://).
* For general Q&A, ideas, or other things, please visit [Discussions](https://).
* For software-related bugs, issues, or suggestions, please use [Issues](https://).


First steps
-----------
`Regularized-SpringRank` _will be_ on PyPI. To start, hit this command on the shell:

```sh
$ pip install regularized-springrank
```

In your Python console, `simplicial-test` is invoked using:

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
TODO.

Related links
-------------
TODO.

Acknowledgement
---------------
TODO.
