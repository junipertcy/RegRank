[![license](https://img.shields.io/badge/license-LGPL-green.svg?style=flat)](https://github.com/junipertcy/regrank/blob/main/LICENSE) [![PyPI version](https://img.shields.io/pypi/v/regrank.svg)](https://pypi.org/project/regrank/) [![PyPI downloads](https://img.shields.io/pypi/dm/regrank.svg?label=Pypi%20downloads)](https://pypi.org/project/regrank/) [![Build Status](https://github.com/junipertcy/regrank/actions/workflows/release.yml/badge.svg)](https://github.com/junipertcy/regrank/actions) [![Open Source Love](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://github.com/ellerbrock/open-source-badges/)

**regrank** implements a collection of regularized, convex models (+solvers) that allow the inference of hierarchical structure in a directed network, which exists due to dominance, social status, or prestige. Specifically, this work leverages the time-varying structure and/or the node metadata present in the data set.

This is the software repository behind the paper:

* Tzu-Chi Yen and Stephen Becker, *Regularized methods for efficient ranking in networks*, in preparation.
* For full documentation, please visit [this site](https://docs.netscied.tw/regrank/index.html).
* General Q&A, ideas, or other things, please visit [Discussions](https://github.com/junipertcy/regrank/discussions).
* Software-related bugs, issues, or suggestions, please use [Issues](https://github.com/junipertcy/regrank/issues).

## Installation

**regrank** is available on PyPI. It also depends on `graph-tool`. We recommend using `conda` to manage packages.

```bash
conda create --name regrank-dev -c conda-forge graph-tool
conda activate regrank-dev
pip install regrank
```

## Example

```python
# Import the library
import regrank as rr

# Load a data set
g = rr.datasets.us_air_traffic()

# Create a model
model = rr.optimize.regrank(method="annotated")

# Fit the model: We decided to analyze the `state_abr` nodal metadata,
# We may inspect `g.list_properties()` for other metadata to analyze.
result = model.fit(g, alpha=1, lambd=0.5, goi="state_abr")

# Now, result["primal"] should have the rankings. We can compute a summary.
summary = rr.compute_summary(g, "state_abr", primal_s=result["primal"])
```

Let's plot the rankings, via `rr.plot_hist(summary)`. Note that most of the node categories are regularized to have the same mean ranking.

![A histogram of four ranking groups, where most of the metadata share the same mean ranking.](docs/assets/us_air_traffic_hist.png)

We provided a summary via `rr.print_summary_table(summary)`.

      +-------+-------+--------+-----------------------------------------+--------+---------+
      | Group | #Tags | #Nodes | Members                                 |   Mean |     Std |
      +-------+-------+--------+-----------------------------------------+--------+---------+
      | 1     |     5 |    825 | CA, WA, OR, TT, AK                      |  0.047 | 1.1e-02 |
      | 2     |     4 |    206 | TX, MT, PA, ID                          | -0.006 | 4.2e-03 |
      | 3     |    43 |   1243 | MI, IN, TN, NC, VA, IL, CO, WV, MA, WI, | -0.035 | 4.3e-03 |
      |       |       |        | SC, KY, MO, MD, AZ, PR, LA, UT, MN, GA, |        |         |
      |       |       |        | MS, HI, DE, NM, ME, NJ, NE, VT, CT, SD, |        |         |
      |       |       |        | IA, NV, ND, AL, OK, AR, NH, RI, OH, FL, |        |         |
      |       |       |        | KS, NY, WY                              |        |         |
      | 4     |     1 |      4 | VI                                      | -0.072 | 0.0e+00 |
      +-------+-------+--------+-----------------------------------------+--------+---------+

The result suggests that states such as `CA`, `WA`, or `AK` are significantly more *popular* than other states.

## Data sets

We have a companion repo—[regrank-data](https://github.com/junipertcy/regrank-data)—for data sets used in the paper. Which are:

* [PhD_exchange](https://github.com/junipertcy/regrank-data/tree/main/PhD_exchange)
* [Parakeet](https://github.com/junipertcy/regrank-data/tree/main/parakeet)

In addendum, we have provided the [regrank.datasets](https://junipertcy.github.io/regrank/datasets.html) submodule to load data sets hosted by other repositories, such as the [Netzschleuder](http://networkrepository.com/). See the docs for more information.

## Development

The library uses pytest to ensure correctness. The test suite depends on [mosek](https://www.mosek.com/) and [gurobi](https://www.gurobi.com/).

## License

**regrank** is open-source and licensed under the [GNU Lesser General Public License v3.0](https://www.gnu.org/licenses/lgpl-3.0.en.html).
