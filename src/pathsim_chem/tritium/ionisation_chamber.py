#########################################################################################
##
##                            Ionisation Chamber Block
##
#########################################################################################

# IMPORTS ===============================================================================

from pathsim.blocks.function import Function

# BLOCKS ================================================================================

class IonisationChamber(Function):
    """Ionisation chamber for tritium detection.

    Algebraic block that models a flow-through ionisation chamber. The sample
    passes through unchanged while the chamber produces a signal proportional
    to the tritium concentration, scaled by a detection efficiency.

    Mathematical Formulation
    -------------------------
    The chamber receives a tritium flux and flow rate, computes the
    concentration, and applies the detection efficiency:

    .. math::

        c = \\frac{\\Phi_{in}}{\\dot{V}}

    .. math::

        \\text{signal} = \\varepsilon(c) \\cdot c

    .. math::

        \\Phi_{out} = \\Phi_{in}

    where :math:`\\varepsilon` is the detection efficiency (constant or
    concentration-dependent).

    Parameters
    ----------
    detection_efficiency : float or callable, optional
        Constant efficiency factor or a function ``f(c) -> float`` that
        returns the efficiency for a given concentration. Mutually
        exclusive with *detection_threshold*.
    detection_threshold : float, optional
        If provided, the efficiency is a step function: 1 above the
        threshold, 0 below. Mutually exclusive with *detection_efficiency*.
    """

    input_port_labels = {
        "flux_in":   0,
        "flow_rate": 1,
    }

    output_port_labels = {
        "flux_out": 0,
        "signal":   1,
    }

    def __init__(self, detection_efficiency=None, detection_threshold=None):

        # input validation
        if detection_efficiency is not None and detection_threshold is not None:
            raise ValueError(
                "Specify either 'detection_efficiency' or 'detection_threshold', not both"
            )
        if detection_efficiency is None and detection_threshold is None:
            raise ValueError(
                "One of 'detection_efficiency' or 'detection_threshold' must be provided"
            )

        if detection_threshold is not None:
            self.detection_efficiency = lambda c: 1.0 if c >= detection_threshold else 0.0
        else:
            self.detection_efficiency = detection_efficiency

        self.detection_threshold = detection_threshold

        super().__init__(func=self._eval)

    def _eval(self, flux_in, flow_rate):
        concentration = flux_in / flow_rate if flow_rate > 0 else 0.0

        eff = self.detection_efficiency
        epsilon = eff(concentration) if callable(eff) else eff

        signal = epsilon * concentration
        flux_out = flux_in

        return (flux_out, signal)
