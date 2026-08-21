# Benchmark specification

## Scientific task

Each benchmark case contains one frequency-time observation of a fixed
three-hour observing configuration. Inference jointly accounts for 30 regional
foreground spectral indices, six continuous radio-galaxy signal parameters,
one Gaussian-noise amplitude, and 100 linear beam coefficients. The beam
coefficients may either be sampled explicitly or integrated out analytically.

The package deliberately exposes no data-generation controls. Changing the
frequency grid, observing window, foreground regions, signal model, or beam
ensemble defines a different benchmark.

## Packaged observations

The package contains 100 directories under
`global21cm_benchmark/simulated_data/dataset_XX`. Each contains:

| File | Shape | Contents |
| --- | --- | --- |
| `observation_XX.npy` | `(86, 37)` | Frequency-time observation in K |
| `signal_XX.npy` | `(86,)` | Injected global 21-cm profile in K |
| `parameters_XX.npy` | `(6,)` | Injected continuous signal parameters |

Use `Dataset(index)` rather than constructing package-data paths:

```python
from global21cm_benchmark import Dataset

dataset = Dataset(0)
observation = dataset.observation
```

`dataset.injected_signal` and `dataset.injected_parameters` are retained for
benchmark evaluation but are never supplied to the likelihood.

## Signal emulator

`evaluate_21cm(parameters)` accepts six continuous parameters in this order:

```text
log10fstarII, log10fstarIII, log10Vc, log10fX, tau, log10fradio
```

It returns an `(86,)` signal in K on the fixed 50--135 MHz grid. The discrete
simulation coordinates are fixed internally to `alpha=1.3`, `nu_0=500 eV`,
and `pop=232`.

## Response operator

`inference_tensors/response_operator.npy` has shape `(86, 37, 101, 30)` with
axes frequency, time, response channel, and foreground region. Channel zero is
the mean-beam response. Channels 1--100 are beam modes multiplied by their
prior standard deviations.

For regional spectral indices $\boldsymbol{\beta}$, the forward model computes

$$
X_{ftk}(\boldsymbol{\beta}) =
\sum_r \mathcal{R}_{ftkr}
\left(\frac{\nu_f}{230\,\mathrm{MHz}}\right)^{-\beta_r}.
$$

The mean foreground is $X_{ft0}$ and the remaining channels form the linear
beam design matrix $H_{ftm}=X_{ft,m+1}$. The explicit model is

$$
m_{ft} = X_{ft0} + \sum_m H_{ftm}z_m + T_{21,f},
\qquad \mathbf{z}\sim\mathcal{N}(\mathbf{0},\mathbf{I}).
$$

`beam_prior.npy` retains the original PCA-score means and standard deviations;
the response operator itself uses the prior-whitened coordinates $\mathbf{z}$.

## Parameter orderings

The 37-dimensional marginalised position is

```text
beta_00, ..., beta_29,
log10fstarII, log10fstarIII, log10Vc, log10fX, tau, log10fradio,
log_noise
```

where `log_noise` is the base-10 logarithm of the Gaussian-noise standard
deviation in K. The 137-dimensional explicit position inserts
`beam_000, ..., beam_099` between the foreground and signal blocks.

The exact names and bounds are immutable benchmark data. `Prior()` loads them
internally: `sample_collapsed` and `sample_full` draw the corresponding
parameterisation, while `evaluate_collapsed` and `evaluate_full` return its
normalized log density. Users do not need to manipulate prior-bound arrays.

## Likelihoods

`Likelihood(dataset).evaluate_full(parameters)` evaluates independent Gaussian
noise around the explicit 137-dimensional forward model.

`Likelihood(dataset).evaluate_collapsed(parameters)` integrates the 100 beam
coefficients out exactly. For residual $\mathbf{r}$, noise variance
$\sigma_n^2$, and flattened design matrix $\mathbf{H}$, it operates in beam
space through

$$
\mathbf{Q}=\mathbf{I}+\sigma_n^{-2}\mathbf{H}^{\mathsf T}\mathbf{H},
\qquad
\mathbf{h}=\sigma_n^{-2}\mathbf{H}^{\mathsf T}\mathbf{r}.
$$

The implemented log likelihood is

$$
\log\mathcal{L}_{\mathrm{coll}}=-\frac12\left[
\sigma_n^{-2}\mathbf{r}^{\mathsf T}\mathbf{r}
-\mathbf{h}^{\mathsf T}\mathbf{Q}^{-1}\mathbf{h}
+N\log\sigma_n^2+\log|\mathbf{Q}|+N\log(2\pi)\right].
$$

This avoids constructing or factorising the dense effective data covariance;
only the 100-by-100 matrix $\mathbf{Q}$ is factorised.

## Public interface

```python
import jax

from global21cm_benchmark import Dataset, Likelihood, Posterior, Prior

dataset = Dataset(0)
likelihood = Likelihood(dataset)
prior = Prior()
posterior = Posterior(dataset)

collapsed_parameters = prior.sample_collapsed(jax.random.key(0))[0]
full_parameters = prior.sample_full(jax.random.key(1))[0]

likelihood.evaluate_collapsed(collapsed_parameters)
likelihood.evaluate_full(full_parameters)
prior.evaluate_collapsed(collapsed_parameters)
prior.evaluate_full(full_parameters)
posterior.evaluate_collapsed(collapsed_parameters)
posterior.evaluate_full(full_parameters)
```

The posterior methods return the corresponding normalized log prior plus log
likelihood. The standalone `forward_model(beta, beam_scores,
signal_parameters)` evaluates the single explicit physical model; analytical
beam marginalisation is a likelihood operation, not a second forward model.

## Nested Slice Sampling

BlackJAX NSS requires the prior and likelihood separately because it samples
from the prior subject to a likelihood constraint. For the collapsed problem:

```python
import blackjax
import jax

from global21cm_benchmark import Dataset, Likelihood, Prior

dataset = Dataset(0)
likelihood = Likelihood(dataset)
prior = Prior()

algorithm = blackjax.nss(
    logprior_fn=prior.evaluate_collapsed,
    loglikelihood_fn=likelihood.evaluate_collapsed,
    num_delete=185,
    num_inner_steps=296,
)

key, initial_key, step_key = jax.random.split(jax.random.key(430000), 3)
live_points = prior.sample_collapsed(initial_key, count=925)
state = algorithm.init(live_points)
state, info = jax.jit(algorithm.step)(step_key, state)
```

Each call to `algorithm.step` performs one complete delete-and-replace
transition, including all configured stepping-out and shrinkage evaluations.
The reference CLI repeats this transition to the evidence stopping criterion,
then performs finalisation, evidence calculation, and likelihood-call
accounting. Finalisation also stores the normalized posterior weights and
automatically creates `signal_corner.png` and `signal_recovery.png` beside the
chain archive.

## Implementation modules

| Module | Purpose |
| --- | --- |
| `global21cm_benchmark.api` | Public dataset and density objects |
| `global21cm_benchmark.analysis` | Weighted corner and signal-recovery plots |
| `global21cm_benchmark.emulator` | Fixed signal emulator |
| `global21cm_benchmark.forward_model` | Packaged data and response contractions |
| `global21cm_benchmark.likelihood` | Collapsed and explicit density calculations |
| `global21cm_benchmark.inference` | Reproduce the reference Nested Slice Sampling run |
