regrank.regularizers.regularizers
=================================

.. py:module:: regrank.regularizers.regularizers


Classes
-------

.. autoapisummary::

   regrank.regularizers.regularizers.Regularizer
   regrank.regularizers.regularizers.zero_reg
   regrank.regularizers.regularizers.same_mean_reg


Module Contents
---------------

.. py:class:: Regularizer(lambd=1)

   Inputs:
       lambd (scalar > 0): regularization coefficient. Default value is 1.
   All regularizers implement the following functions:
   1. evaluate(theta). Evaluates the regularizer at theta.
   2. prox(t, nu, warm_start, pool): Evaluates the proximal operator of the regularizer at theta


   .. py:attribute:: lambd
      :value: 1



   .. py:method:: evaluate(theta)
      :abstractmethod:



   .. py:method:: prox(t, nu, warm_start, pool)
      :abstractmethod:



.. py:class:: zero_reg(lambd=1)

   Bases: :py:obj:`Regularizer`


   Inputs:
       lambd (scalar > 0): regularization coefficient. Default value is 1.
   All regularizers implement the following functions:
   1. evaluate(theta). Evaluates the regularizer at theta.
   2. prox(t, nu, warm_start, pool): Evaluates the proximal operator of the regularizer at theta


   .. py:attribute:: lambd
      :value: 1



   .. py:method:: evaluate(theta)


   .. py:method:: prox(t, nu, warm_start, pool)


.. py:class:: same_mean_reg(lambd=1)

   Bases: :py:obj:`Regularizer`


   Inputs:
       lambd (scalar > 0): regularization coefficient. Default value is 1.
   All regularizers implement the following functions:
   1. evaluate(theta). Evaluates the regularizer at theta.
   2. prox(t, nu, warm_start, pool): Evaluates the proximal operator of the regularizer at theta


   .. py:attribute:: lambd
      :value: 1



   .. py:method:: evaluate(theta)

      Indicate if the input 'theta' is in the constraint set or not

      :param theta: input value (in the dual space)
      :type theta: float

      :returns: 0 if in the constraint set, +inf otherwise
      :rtype: float



   .. py:method:: evaluate_cvx(theta)


   .. py:method:: prox(theta, t)
