regrank.utils.stats.experiments
===============================

.. py:module:: regrank.utils.stats.experiments


Attributes
----------

.. autoapisummary::

   regrank.utils.stats.experiments.username


Classes
-------

.. autoapisummary::

   regrank.utils.stats.experiments.Experiment
   regrank.utils.stats.experiments.PhDExchange
   regrank.utils.stats.experiments.PeerInstitution


Module Contents
---------------

.. py:data:: username
   :value: 'tzuchi'


.. py:class:: Experiment

   .. py:attribute:: basic_stats


   .. py:method:: get_data() -> graph_tool.all.Graph


   .. py:method:: draw()


   .. py:method:: print_sorted_mean(num=5, precision=3)


.. py:class:: PhDExchange

   Bases: :py:obj:`Experiment`


   .. py:attribute:: g


   .. py:attribute:: num_classes
      :value: 0



   .. py:attribute:: num_dual_vars
      :value: 0



   .. py:attribute:: num_primal_vars
      :value: 0



   .. py:attribute:: data_goi
      :value: None



   .. py:attribute:: basic_stats


   .. py:method:: get_data(goi='sector') -> graph_tool.all.Graph

      goi: which stratum (metadata of the nodes) that you are looking for?




   .. py:method:: get_node_metadata()


   .. py:method:: draw()


   .. py:method:: plot_hist(bin_count=20, legend=False)


   .. py:method:: compute_basic_stats(sslc=None, dual_v=None, primal_s=None)


   .. py:method:: print_sorted_diff(num=5)


.. py:class:: PeerInstitution

   Bases: :py:obj:`Experiment`


   .. py:attribute:: g


   .. py:attribute:: num_classes
      :value: 0



   .. py:attribute:: num_dual_vars
      :value: 0



   .. py:attribute:: num_primal_vars
      :value: 0



   .. py:attribute:: data_goi
      :value: None



   .. py:attribute:: basic_stats


   .. py:method:: get_data(goi='sector') -> graph_tool.all.Graph

      goi: which stratum (metadata of the nodes) that you are looking at?




   .. py:method:: compute_basic_stats(sslc=None, dual_v=None, primal_s=None)


   .. py:method:: get_node_metadata()


   .. py:method:: draw()


   .. py:method:: plot_hist(bin_count=20, legend=False, saveto=None)


   .. py:method:: print_sorted_diff(num=5)
