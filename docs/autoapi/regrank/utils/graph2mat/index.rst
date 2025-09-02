regrank.utils.graph2mat
=======================

.. py:module:: regrank.utils.graph2mat


Functions
---------

.. autoapisummary::

   regrank.utils.graph2mat.cast2sum_squares_form_t
   regrank.utils.graph2mat.cast2sum_squares_form
   regrank.utils.graph2mat.compute_cache_from_data_t
   regrank.utils.graph2mat.compute_cache_from_data
   regrank.utils.graph2mat.compute_Bt_B_inv
   regrank.utils.graph2mat.grad_g_star
   regrank.utils.graph2mat.filter_by_year
   regrank.utils.graph2mat.compute_ell


Module Contents
---------------

.. py:function:: cast2sum_squares_form_t(g, alpha, lambd, from_year=1960, to_year=1961, top_n=70, separate=False)

   Operator to linearize the sum of squares loss function.

   :param g: _description_
   :type g: _type_
   :param alpha: _description_
   :type alpha: _type_
   :param lambd: _description_
   :type lambd: _type_
   :param from_year: _description_. Defaults to 1960.
   :type from_year: int, optional
   :param to_year: _description_. Defaults to 1961.
   :type to_year: int, optional
   :param top_n: _description_. Defaults to 70.
   :type top_n: int, optional
   :param separate: _description_. Defaults to False.
   :type separate: bool, optional

   :raises ValueError: _description_
   :raises ValueError: _description_
   :raises TypeError: _description_

   :returns: _description_
   :rtype: _type_


.. py:function:: cast2sum_squares_form(data, alpha, regularization=True)

   This is how we linearize the objective function:
   B_ind  i  j
   0      0  1
   1      0  2
   2      0  3
   3      1  0
   4      1  2
   5      1  3
   6      2  0
   ...
   11     3  2
   12     0  0
   13     1  1
   14     2  2
   15     3  3


.. py:function:: compute_cache_from_data_t(data, alpha=1, lambd=1, from_year=1960, to_year=1961, top_n=70)

.. py:function:: compute_cache_from_data(data, alpha, regularization=True, **kwargs)

   _summary_

   Args:

   data (_type_): _description_

   alpha (_type_): _description_

   regularization (bool, optional): _description_. Defaults to True.

   Returns:

   dictionary: _description_



.. py:function:: compute_Bt_B_inv(B)

.. py:function:: grad_g_star(B, b, v)

.. py:function:: filter_by_year(g, from_year=1946, to_year=2006, top_n=70)

.. py:function:: compute_ell(g, key=None)
