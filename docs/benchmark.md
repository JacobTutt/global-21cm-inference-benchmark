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

Use `load_dataset(index)` rather than constructing package-data paths:

```python
from global21cm_benchmark.forward_model import load_dataset

observation, injected_signal, injected_parameters = load_dataset(0)
```

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

The bounds and exact names are stored in `parameter_priors.npz` and exposed as
`PRIOR_LOWER`, `PRIOR_UPPER`, and `PARAMETER_NAMES` from
`global21cm_benchmark.likelihood`.

## Likelihoods

`full_log_likelihood(parameters, observation)` evaluates independent Gaussian
noise around the explicit 137-dimensional forward model.

`collapsed_log_likelihood(parameters, observation)` integrates the 100 beam
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

## Public modules

| Module | Purpose |
| --- | --- |
| `global21cm_benchmark.emulator` | Evaluate the fixed signal emulator |
| `global21cm_benchmark.forward_model` | Load datasets and evaluate mean or explicit models |
| `global21cm_benchmark.likelihood` | Priors and collapsed or explicit likelihoods |
| `global21cm_benchmark.inference` | Reproduce the reference Nested Slice Sampling run |
