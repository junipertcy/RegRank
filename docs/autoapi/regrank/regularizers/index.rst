regrank.regularizers
====================

.. py:module:: regrank.regularizers


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/regrank/regularizers/annotated/index
   /autoapi/regrank/regularizers/branch_constrained/index
   /autoapi/regrank/regularizers/dictionary/index
   /autoapi/regrank/regularizers/huber/index
   /autoapi/regrank/regularizers/l1time/index
   /autoapi/regrank/regularizers/l2time/index
   /autoapi/regrank/regularizers/legacy/index
   /autoapi/regrank/regularizers/regularizers/index


Classes
-------

.. autoapisummary::

   regrank.regularizers.BaseRegularizer
   regrank.regularizers.RegularizerFactory


Package Contents
----------------

.. py:class:: BaseRegularizer(cfg: omegaconf.DictConfig)

   Bases: :py:obj:`abc.ABC`


   Helper class that provides a standard way to create an ABC using
   inheritance.


   .. py:attribute:: cfg


   .. py:method:: fit(data, cfg: omegaconf.DictConfig) -> dict[str, Any]
      :abstractmethod:



.. py:class:: RegularizerFactory

   .. py:method:: create(cfg: omegaconf.DictConfig) -> BaseRegularizer
      :classmethod:
