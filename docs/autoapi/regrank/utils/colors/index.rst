regrank.utils.colors
====================

.. py:module:: regrank.utils.colors


Functions
---------

.. autoapisummary::

   regrank.utils.colors.rgb_to_hex
   regrank.utils.colors.hex_to_rgb_normalized
   regrank.utils.colors.generate_complementary_colors
   regrank.utils.colors.generate_adjacent_colors
   regrank.utils.colors.reverse_dict


Module Contents
---------------

.. py:function:: rgb_to_hex(rgb_color)

   Converts an RGB tuple (values in [0,1]) to a hex color string.


.. py:function:: hex_to_rgb_normalized(hex_color)

   Converts a hex color string to an RGB tuple (values in [0,1]).


.. py:function:: generate_complementary_colors(k)

   Generates k visually distinct colors by cycling through hues.


.. py:function:: generate_adjacent_colors(hex_color, k)

   Generates a list of k adjacent hex colors near the given hex color.


.. py:function:: reverse_dict(input_dict)

   Reverses a dictionary, grouping keys by their values.
