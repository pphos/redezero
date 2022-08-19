import numbers
import typing as tp
import numpy as np

from redezero import variable


ScalarValue = tp.Union[
    np.generic,
    bytes,
    str,
    int,
    float,
    memoryview,
]

OperandValue = tp.Union[
    variable.Variable,
    np.ndarray
]
