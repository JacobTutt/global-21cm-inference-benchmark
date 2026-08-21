"""Small public interface to the fixed benchmark."""

from .forward_model import forward_model, load_dataset
from .likelihood import (
    _collapsed_log_likelihood,
    _full_log_likelihood,
    _log_prior,
    _sample_prior,
)


class Dataset:
    """Load one observation and retain its injected values for evaluation."""

    def __init__(self, index):
        self.index = index
        self.observation, self.injected_signal, self.injected_parameters = load_dataset(
            index
        )


class Likelihood:
    """Evaluate either likelihood for one fixed :class:`Dataset`."""

    def __init__(self, dataset):
        if not isinstance(dataset, Dataset):
            raise TypeError("Likelihood expects a Dataset instance.")
        self.dataset = dataset

    def evaluate_collapsed(self, parameters):
        """Evaluate the 37-dimensional beam-marginalised log likelihood."""
        return _collapsed_log_likelihood(parameters, self.dataset.observation)

    def evaluate_full(self, parameters):
        """Evaluate the 137-dimensional explicit log likelihood."""
        return _full_log_likelihood(parameters, self.dataset.observation)


class Prior:
    """Evaluate or sample the normalized priors for both parameterisations."""

    def evaluate_collapsed(self, parameters):
        """Evaluate the normalized 37-dimensional log prior."""
        return _log_prior(parameters, full=False)

    def evaluate_full(self, parameters):
        """Evaluate the normalized 137-dimensional log prior."""
        return _log_prior(parameters, full=True)

    def sample_collapsed(self, key, count=1):
        """Draw ``count`` independent 37-dimensional prior samples."""
        return _sample_prior(key, count, full=False)

    def sample_full(self, key, count=1):
        """Draw ``count`` independent 137-dimensional prior samples."""
        return _sample_prior(key, count, full=True)


class Posterior:
    """Evaluate prior plus likelihood for one fixed :class:`Dataset`."""

    def __init__(self, dataset):
        self.prior = Prior()
        self.likelihood = Likelihood(dataset)

    def evaluate_collapsed(self, parameters):
        """Evaluate the 37-dimensional unnormalized log posterior."""
        return self.prior.evaluate_collapsed(
            parameters
        ) + self.likelihood.evaluate_collapsed(parameters)

    def evaluate_full(self, parameters):
        """Evaluate the 137-dimensional unnormalized log posterior."""
        return self.prior.evaluate_full(parameters) + self.likelihood.evaluate_full(
            parameters
        )


__all__ = ["Dataset", "Likelihood", "Posterior", "Prior", "forward_model"]
