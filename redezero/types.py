import typing as tp
import numpy as np
import numpy.typing as npt

from redezero import variable


ScalarValue = tp.Union[
    np.generic,
    bytes,
    str,
    int,
    float,
]

OperandValue = tp.Union[
    variable.Variable,
    npt.NDArray
]
