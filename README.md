# Global 21-cm Inference Benchmark

`global21cm-benchmark` is a fixed, end-to-end Bayesian inference benchmark for
global 21-cm cosmology. It packages 100 simulated observations, a neural signal
emulator, a physics-informed foreground model, a linear beam surrogate, and
both explicit and analytically marginalised likelihoods. Expensive sky and
beam calculations are already reduced to immutable inference tensors, so the
benchmark has no dependency on the simulation pipelines that generated them.

## Installation

Python 3.11--3.13 is supported.

```bash
git clone https://github.com/JacobTutt/global-21cm-inference-benchmark.git
cd global-21cm-inference-benchmark
pip install .
```

For NVIDIA GPUs, install the CUDA-enabled JAX extra:

```bash
pip install ".[gpu]"
```

## Quickstart

The benchmark objects expose ordinary JAX-compatible methods:

```python
import jax

from global21cm_benchmark import Dataset, Likelihood, Posterior, Prior

dataset = Dataset(0)
likelihood = Likelihood(dataset)
prior = Prior()
posterior = Posterior(dataset)

parameters = prior.sample_collapsed(jax.random.key(0))[0]
print(jax.jit(likelihood.evaluate_collapsed)(parameters))
print(jax.jit(posterior.evaluate_collapsed)(parameters))
```

Use `evaluate_full` and `sample_full` for the 137-dimensional explicit model.
The injected values are available as `dataset.injected_signal` and
`dataset.injected_parameters` for evaluation only.

## Benchmark

| Quantity | Value |
| --- | ---: |
| Observations | 100 |
| Observation shape | `(86 frequencies, 37 times)` |
| Foreground parameters | 30 |
| Signal parameters | 6 |
| Noise parameters | 1 |
| Beam coefficients | 100 |
| Marginalised dimension | 37 |
| Explicit dimension | 137 |

The marginalised likelihood integrates out all 100 linear beam coefficients
analytically. The explicit likelihood includes their prior-whitened values in
the sampled position. Both use the same emulator, foreground response, noise
model, priors, and observations.

See [Benchmark specification](docs/benchmark.md) for the object interface,
parameter orderings, array shapes, equations, and packaged-data layout.
Function-level details are available through the public classes:

```python
from global21cm_benchmark import Dataset, Likelihood, Posterior, Prior

help(Dataset)
help(Likelihood)
help(Prior)
help(Posterior)
```

## Reference inference

Run the 37-dimensional analytically marginalised benchmark with:

```bash
global21cm-inference 0
```

Run the 137-dimensional explicit comparison with:

```bash
global21cm-inference 0 --full
```

Both use the paper configuration: 25 live points and eight inner slice steps
per dimension, a 20% deletion fraction, and a `dlogZ < -3` stopping criterion.
Results are written to `results/dataset_XX/{marginalised,full}`.

## Validation

Check the archived emulator, forward-model, and likelihood reference values:

```bash
python -m tests.test_reference
```

For the GPU container:

```bash
docker build -t global21cm-benchmark .
docker run --gpus all -v "$PWD/results:/benchmark/results" global21cm-benchmark 0
```
