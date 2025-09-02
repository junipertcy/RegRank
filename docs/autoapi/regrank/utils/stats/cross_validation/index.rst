regrank.utils.stats.cross_validation
====================================

.. py:module:: regrank.utils.stats.cross_validation


Attributes
----------

.. autoapisummary::

   regrank.utils.stats.cross_validation.gt
   regrank.utils.stats.cross_validation.optuna
   regrank.utils.stats.cross_validation.ax_optimize


Classes
-------

.. autoapisummary::

   regrank.utils.stats.cross_validation.HyperparameterTuner
   regrank.utils.stats.cross_validation.GridSearchTuner
   regrank.utils.stats.cross_validation.OptunaTuner
   regrank.utils.stats.cross_validation.AxTuner
   regrank.utils.stats.cross_validation.CrossValidation


Functions
---------

.. autoapisummary::

   regrank.utils.stats.cross_validation.get_cv_realization


Module Contents
---------------

.. py:data:: gt
   :value: None


.. py:data:: optuna
   :value: None


.. py:data:: ax_optimize
   :value: None


.. py:function:: get_cv_realization(graph: graph_tool.all.Graph, n_splits: int, seed: int | None = None) -> dict[int, graph_tool.all.EdgePropertyMap]

   Generates cross-validation splits for the edges of a graph.


.. py:class:: HyperparameterTuner(evaluation_function: collections.abc.Callable, params_grid: dict, n_trials: int = 50)

   Bases: :py:obj:`abc.ABC`


   Abstract base class for hyperparameter tuners.


   .. py:attribute:: evaluation_function


   .. py:attribute:: params_grid


   .. py:attribute:: n_trials
      :value: 50



   .. py:method:: tune() -> tuple[dict[str, float], dict[str, float]]
      :abstractmethod:


      Run the tuning process.



.. py:class:: GridSearchTuner(evaluation_function: collections.abc.Callable, params_grid: dict, interp_grid: dict | None = None, **kwargs)

   Bases: :py:obj:`HyperparameterTuner`


   Performs hyperparameter tuning using grid search with interpolation.


   .. py:attribute:: interp_grid
      :value: None



   .. py:method:: tune() -> tuple[dict[str, float], dict[str, float]]

      Run the tuning process.



.. py:class:: OptunaTuner(evaluation_function: collections.abc.Callable, params_grid: dict, n_trials: int = 50)

   Bases: :py:obj:`HyperparameterTuner`


   Performs hyperparameter tuning using Optuna.


   .. py:method:: tune() -> tuple[dict[str, float], dict[str, float]]

      Run the tuning process.



.. py:class:: AxTuner(evaluation_function: collections.abc.Callable, params_grid: dict, n_trials: int = 50)

   Bases: :py:obj:`HyperparameterTuner`


   Performs hyperparameter tuning using Ax.


   .. py:method:: tune() -> tuple[dict[str, float], dict[str, float]]

      Run the tuning process.



.. py:class:: CrossValidation(g: graph_tool.all.Graph, n_folds: int = 5, n_subfolds: int = 4, n_reps: int = 3, seed: int = 42, goi: Any = None)

   Performs cross-validation to find optimal hyperparameters.


   .. py:attribute:: TUNER_MAP


   .. py:attribute:: g


   .. py:attribute:: n_folds
      :value: 5



   .. py:attribute:: n_subfolds
      :value: 4



   .. py:attribute:: n_reps
      :value: 3



   .. py:attribute:: seed
      :value: 42



   .. py:attribute:: goi
      :value: None



   .. py:attribute:: model
      :type:  Any | None
      :value: None



   .. py:attribute:: main_cv_splits
      :type:  dict[int, graph_tool.all.EdgePropertyMap]


   .. py:attribute:: sub_cv_splits
      :type:  dict[int, dict[int, dict[int, graph_tool.all.EdgePropertyMap]]]


   .. py:attribute:: cv_results
      :type:  dict[str, dict[str, dict[int, Any]]]


   .. py:method:: prepare_cv_splits()


   .. py:method:: train_and_validate(model: Any, fold_id: int, tuner_type: str, params_grid: dict, **kwargs)
