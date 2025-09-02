regrank.utils.NamedGraph
========================

.. py:module:: regrank.utils.NamedGraph


Classes
-------

.. autoapisummary::

   regrank.utils.NamedGraph.NamedGraph


Module Contents
---------------

.. py:class:: NamedGraph(directed=True)

   .. py:attribute:: g


   .. py:attribute:: name_prop


   .. py:attribute:: name_to_vertex


   .. py:attribute:: vertex_counter
      :value: 0



   .. py:method:: add_vertex(name)

      Add a vertex with a given name if it doesn't exist.



   .. py:method:: add_edge(source_name, target_name, weight=1.0)

      Add an edge between two named vertices.



   .. py:method:: assign_ranking(names, rankings)

      Assign true rankings to nodes.

      Parameters:
      - names: Either a string (single node name) or list of strings (node names)
      - rankings: Either a float (single ranking) or list of floats (rankings)



   .. py:method:: get_ranking(name)

      Get the true ranking of a node by name.



   .. py:method:: list_rankings()

      List all nodes with their true rankings.



   .. py:method:: get_vertex_by_name(name)

      Get vertex descriptor by name.



   .. py:method:: get_name_by_vertex(vertex)

      Get name by vertex descriptor.



   .. py:method:: get_vertex_index(name)

      Get the integer index of a vertex by name.



   .. py:method:: list_vertices()

      List all vertices with their names and indices.



   .. py:method:: list_edges()

      List all edges with names.



   .. py:property:: graph

      Access the underlying graph_tool Graph object.
