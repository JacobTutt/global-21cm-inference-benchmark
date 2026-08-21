# Global 21-cm Inference Benchmark

This repository contains the fixed end-to-end inference problem used for the
radio-galaxy signal-recovery benchmark. It is intentionally not a configurable
analysis pipeline: sky, beam and signal simulations have already been reduced
to the tensors required by inference.

## Data contract

Each `simulated_data/dataset_XX` directory contains:

- `observation_XX.npy`: the `(86, 37)` frequency-time observation in K;
- `signal_XX.npy`: the injected 21-cm profile in K;
- `parameters_XX.npy`: its six continuous astrophysical parameters.

`inference_tensors/response_operator.npy` has shape `(86, 37, 101, 30)`.
Channel zero is the mean-beam response and the remaining 100 channels are
beam-prior-whitened response modes. For spectral indices `beta`, inference
constructs

```text
terms[f,t,k] = sum_r response[f,t,k,r] * (frequency[f] / 230 MHz) ** (-beta[r])
```

The explicit model samples the whitened beam scores from `Normal(0, 1)`. The
marginalised likelihood integrates them out exactly. `beam_prior.npy` records
the original PCA-score prior as `[mean, standard deviation]` for each mode.

`forward_model.py` exposes the mean and explicit forward models.
`likelihood.py` exposes `collapsed_log_likelihood` and `full_log_likelihood`.

The only signal model is `emulator.evaluate_21cm(parameters)`. It accepts the
six parameters listed in `emulator.PARAMETER_NAMES`; the discrete simulation
coordinates are fixed internally to `alpha=1.3`, `nu_0=500 eV`, and `pop=232`.

## Run

```bash
pip install -r requirements.txt
python run_inference.py 0
```

The default run analytically marginalises the 100 beam coefficients. The
137-dimensional explicit comparison is:

```bash
python run_inference.py 0 --full
```

Both use the paper configuration: 25 live points per dimension, eight inner
slice steps per dimension, a 20% deletion fraction and `dlogZ < -3` stopping.

Check the archived numerical contract with:

```bash
python -m tests.test_reference
```

For a GPU container:

```bash
docker build -t global21cm-benchmark .
docker run --gpus all -v "$PWD/results:/benchmark/results" global21cm-benchmark 0
```
