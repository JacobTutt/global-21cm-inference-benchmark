"""Public interface for the fixed radio-galaxy signal emulator."""

from .model import FREQUENCIES_MHZ, PARAMETER_NAMES, evaluate_21cm

__all__ = ["FREQUENCIES_MHZ", "PARAMETER_NAMES", "evaluate_21cm"]
