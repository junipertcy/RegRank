.. image:: https://img.shields.io/badge/license-LGPL-green.svg?style=flat
   :target: https://github.com/junipertcy/rSpringRank/blob/main/LICENSE

.. image:: https://img.shields.io/pypi/dm/rSpringRank.svg?label=Pypi%20downloads
   :target: https://pypi.org/project/rSpringRank/

**rSpringRank** implements a collection of regularized, convex models (+solvers) that allow the inference of hierarchical structure in a directed network, which exists due to dominance, social status, or prestige. Specifically, this work leverages the time-varying structure and/or the node metadata present in the data set.

This is the software repository behind the paper:

* Tzu-Chi Yen and Stephen Becker, *Regularized methods for efficient ranking in networks*, in preparation.

* For full documentation, please visit `this site <https://docs.netscied.tw/rSpringRank/index.html>`_.

* General Q&A, ideas, or other things, please visit `Discussions <https://github.com/junipertcy/rSpringRank/discussions>`_.

* Software-related bugs, issues, or suggestions, please use `Issues <https://github.com/junipertcy/rSpringRank/issues>`_.

Installation
------------

**rSpringRank** is available on PyPI, so do ``pip install rSpringRank``.


Data sets
---------

We have a companion repo—`rSpringRank-data <https://github.com/junipertcy/rSpringRank-data>`_—for data sets used in the paper. Which are:

* `PhD_exchange <https://github.com/junipertcy/rSpringRank-data/tree/main/PhD_exchange>`_
* `Parakeet <https://github.com/junipertcy/rSpringRank-data/tree/main/parakeet>`_

In addendum, we have provided the `rSpringRank.datasets <https://junipertcy.github.io/rSpringRank/datasets.html>`_ 
submodule to load data sets hosted by other repositories, such as the `Netzschleuder <http://networkrepository.com/>`. 
See the docs for more information.

Development
-----------

The library uses pytest to ensure correctness. The test suite depends on `mosek <https://www.mosek.com/>`_ and `gurobi <https://www.gurobi.com/>`_.

License
-------

**rSpringRank** is open-source and licensed under the `GNU Lesser General Public License v3.0 <https://www.gnu.org/licenses/lgpl-3.0.en.html>`_.
