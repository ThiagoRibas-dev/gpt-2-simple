from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .gpt_2 import *

"""TensorFlow Graph Editor."""
# pylint: disable=wildcard-import

from gpt_2_simple.src.tensorflow.contrib.graph_editor.edit import *
from gpt_2_simple.src.tensorflow.contrib.graph_editor.reroute import *
from gpt_2_simple.src.tensorflow.contrib.graph_editor.select import *
from gpt_2_simple.src.tensorflow.contrib.graph_editor.subgraph import *
from gpt_2_simple.src.tensorflow.contrib.graph_editor.transform import *
from gpt_2_simple.src.tensorflow.contrib.graph_editor.util import *

# pylint: enable=wildcard-import

# some useful aliases
# pylint: disable=g-bad-import-order
from gpt_2_simple.src.tensorflow.contrib.graph_editor import subgraph as _subgraph
from gpt_2_simple.src.tensorflow.contrib.graph_editor import util as _util
# pylint: enable=g-bad-import-order
ph = _util.make_placeholder_from_dtype_and_shape
sgv = _subgraph.make_view
sgv_scope = _subgraph.make_view_from_scope

del absolute_import
del division
del print_function
