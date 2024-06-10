.. image:: https://img.shields.io/badge/license-LGPL-green.svg?style=flat
    :target: https://github.com/junipertcy/rSpringRank/blob/main/LICENSE

.. image:: https://img.shields.io/pypi/dm/rSpringRank.svg?label=Pypi%20downloads
  :target: https://pypi.org/project/rSpringRank/

**rSpringRank** implements a collection of regularized, convex models (+solvers) that allow the inference of hierarchical structure in a directed network, which exists due to dominance, social status, or prestige. Specifically, this work leverages the time-varying structure and/or the node metadata present in the data set.

This is the software repository behind the paper:
* Tzu-Chi Yen and Stephen Becker, *Regularized methods for efficient ranking in networks*, in preparation.


* For full documentation, please visit [this site](https://).
* For general Q&A, ideas, or other things, please visit [Discussions](https://).
* For software-related bugs, issues, or suggestions, please use [Issues](https://).


Installation
------------
**rSpringRank** is available on PyPI, so do `pip install rSpringRank`.

Development
-----------

The library uses pytest to ensure correctness. The test suite depends on [mosek](https://www.mosek.com/) and [gurobi](https://www.gurobi.com/).

License
-------
**rSpringRank** is open-source and licensed under the [GNU Lesser General Public License v3.0](https://www.gnu.org/licenses/lgpl-3.0.en.html).
