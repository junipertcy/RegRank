regrank.solvers.cvx
===================

.. py:module:: regrank.solvers.cvx


Classes
-------

.. autoapisummary::

   regrank.solvers.cvx.same_mean_cvx
   regrank.solvers.cvx.legacy_cvx
   regrank.solvers.cvx.huber_cvx


Module Contents
---------------

.. py:class:: same_mean_cvx(g, L, **kwargs)

   .. py:attribute:: g


   .. py:attribute:: L


   .. py:attribute:: sslc
      :value: None



   .. py:attribute:: ssl
      :value: None



   .. py:attribute:: goi


   .. py:method:: loss_fn(dual_v)


   .. py:method:: objective_fn(dual_v)


   .. py:method:: loss_fn_primal(primal_s, alpha=1)


   .. py:method:: regularizer(primal_s)


   .. py:method:: objective_fn_primal(primal_s, lambd=1)


.. py:class:: legacy_cvx(g, alpha=1)

   .. py:attribute:: g


   .. py:attribute:: alpha
      :value: 1



   .. py:attribute:: ssl


   .. py:method:: loss_fn_primal(primal_s)


   .. py:method:: objective_fn_primal(primal_s)


.. py:class:: huber_cvx(g, alpha=1, M=1, incl_reg=False)

   .. py:attribute:: g


   .. py:attribute:: alpha
      :value: 1



   .. py:attribute:: M
      :value: 1



   .. py:attribute:: hl


   .. py:method:: loss_fn_primal(primal_s)


   .. py:method:: objective_fn_primal(primal_s)
