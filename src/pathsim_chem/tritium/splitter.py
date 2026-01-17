#########################################################################################
##
##                                 Flow Splitter Block 
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from pathsim.blocks.function import Function
from pathsim.utils.register import Register


# BLOCKS ================================================================================

class Splitter(Function):
    """Splitter block that splits the input signal into multiple 
    outputs weighted with the specified fractions.

    Note
    ----
    The output fractions must sum to one.
    
    Parameters
    ----------
    fractions : np.ndarray | list
        fractions to split the input signal into, 
        must sum up to one
    """

    input_port_labels = {"in": 0}
    output_port_labels = None

    def __init__(self, fractions=None):

        self.fractions = np.ones(1) if fractions is None else np.array(fractions)

        # input validation
        if not np.isclose(sum(self.fractions), 1):
            raise ValueError(f"'fractions' must sum to one and not {sum(self.fractions)}")

        # initialize like `Function` block
        super().__init__(func=lambda u: self.fractions*u)