from .ops import *
from .core.tensor import Tensor
from .backend_ndarray.ndarray import cpu, cuda
from . import init
from .init import ones, zeros, zeros_like, ones_like

from . import core
from . import data
from . import init
from . import nn
from . import ops
from . import optim