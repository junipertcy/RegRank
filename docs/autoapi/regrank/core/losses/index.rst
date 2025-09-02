regrank.core.losses
===================

.. py:module:: regrank.core.losses


Classes
-------

.. autoapisummary::

   regrank.core.losses.Loss
   regrank.core.losses.huber_loss
   regrank.core.losses.sum_squared_loss
   regrank.core.losses.sum_squared_loss_conj


Module Contents
---------------

.. py:class:: Loss

   .. py:method:: evaluate(theta)
      :abstractmethod:



   .. py:method:: setup(data, K)
      :abstractmethod:


      This function has any important setup required for the problem.



   .. py:method:: prox(t, nu, data, warm_start, pool)
      :abstractmethod:



   .. py:method:: anll(data, G)


.. py:class:: huber_loss

   Bases: :py:obj:`Loss`


   TODO


   .. py:attribute:: B
      :value: None



   .. py:attribute:: b
      :value: None



   .. py:attribute:: M
      :value: 0



   .. py:method:: evaluate_cvx(theta)


   .. py:method:: setup(data, alpha, M, incl_reg)

      This function has any important setup required for the problem.



.. py:class:: sum_squared_loss

   Bases: :py:obj:`Loss`


   f(s) = || B @ s - b ||_2^2


   .. py:attribute:: B
      :value: None



   .. py:attribute:: b
      :value: None



   .. py:attribute:: ell
      :value: None



   .. py:attribute:: Bt_B_inv
      :value: None



   .. py:method:: evaluate(theta)


   .. py:method:: evaluate_cvx(theta)


   .. py:method:: setup(data, alpha, **kwargs)

      This function has any important setup required for the problem.



   .. py:method:: prox(theta)
      :abstractmethod:



   .. py:method:: dual2primal(v)
      :abstractmethod:



   .. py:method:: predict()


   .. py:method:: scores()


   .. py:method:: logprob()


.. py:class:: sum_squared_loss_conj

   Bases: :py:obj:`Loss`


   Conjugate of ...
   f(s) = || B @ s - b ||_2^2


   .. py:attribute:: B
      :value: None



   .. py:attribute:: b
      :value: None



   .. py:attribute:: ell
      :value: None



   .. py:attribute:: Bt_B_inv
      :value: None



   .. py:attribute:: Bt_B_invSqrt
      :value: None



   .. py:attribute:: Bt_B_invSqrt_Btb
      :value: None



   .. py:attribute:: Bt_B_invSqrt_ellt
      :value: None



   .. py:attribute:: ell_BtB_inv_Bt_b
      :value: None



   .. py:attribute:: ell_BtB_inb_ellt
      :value: None



   .. py:attribute:: term_2
      :value: None



   .. py:method:: find_Lipschitz_constant()


   .. py:method:: evaluate(theta)


   .. py:method:: evaluate_cvx(theta)


   .. py:method:: setup(data, alpha, **kwargs)

      This function has any important setup required for the problem.



   .. py:method:: prox(theta)


   .. py:method:: dual2primal(v)


   .. py:method:: predict()


   .. py:method:: scores()


   .. py:method:: logprob()
