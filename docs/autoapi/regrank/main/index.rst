regrank.main
============

.. py:module:: regrank.main


Attributes
----------

.. autoapisummary::

   regrank.main.logger


Classes
-------

.. autoapisummary::

   regrank.main.SpringRank


Module Contents
---------------

.. py:data:: logger

.. py:class:: SpringRank(cfg: omegaconf.DictConfig)

   Bases: :py:obj:`regrank.core.base.BaseModel`


   The main user-facing class for the Regularized-SpringRank library.

   This class acts as a dispatcher, using a configuration object to instantiate
   and run the appropriate regularizer (e.g., legacy, annotated, dictionary).

   Example usage with Hydra:

   @hydra.main(config_path="conf", config_name="config")
   def my_app(cfg: DictConfig) -> None:
       model = SpringRank(cfg)
       results = model.fit(my_graph_data)
       print(results['primal'])


   .. py:attribute:: cfg


   .. py:attribute:: regularizer


   .. py:attribute:: result
      :type:  dict[str, Any]


   .. py:method:: fit(data: graph_tool.all.Graph, **kwargs) -> dict[str, Any]

      Fits the configured SpringRank model to the provided graph data.

      This method delegates the actual fitting process to the specific regularizer
      object that was created during initialization.

      :param data: The graph data, typically a graph_tool.Graph object.
      :param \*\*kwargs: Runtime parameters that can override the initial configuration.
                         This is useful for quick experiments or adjustments without
                         changing config files.

      :returns: A dictionary containing the results of the fitting process, such as
                the primal solution (rankings), final objective value, etc. The exact
                contents depend on the regularizer used.
