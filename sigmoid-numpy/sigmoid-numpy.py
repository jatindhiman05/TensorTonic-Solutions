import numpy as np

def sigmoid(x: list | float) -> np.ndarray | float:
    """
    Returns the sigmoid value for a scalar or each element of a list.
    """
    x = np.array(x)

    result = 1 / (1 + np.exp(-x))

    if result.ndim == 0:
        return float(result)

    return result