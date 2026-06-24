# Markov Chain Monte Carlo with Diffusion Paths

Code accompanying the paper *Markov Chain Monte Carlo with Diffusion Paths*
(paper link: coming soon), introducing the
**Metropolis-adjusted diffusion path (MAD-Path)** sampler — an MCMC algorithm that uses a
trained diffusion model to propose structured, nonlocal moves between isolated regions of
probability mass, accepted/rejected with a Metropolis–Hastings correction so the chain remains
exact.


## Repository structure

```
src/
  target/      Target log-densities for the six examples in Section 6
  sampler/
    vi/        Nonequilibrium DDS (the diffusion model trained as MAD-Path's proposal)
    mcmc/      MAD-Path, adaptive parallel tempering, tempered transitions, MALA/RWM
    importance_sampling/   Adaptive AIS and adaptive SMC baselines
experiments/   One run_*.py (+ matching plot_*.py) per example in Section 6
docs/          Design notes 
```

## Installation

```bash
pip install -r requirements.txt
```

`blackjax` is only needed for the NUTS reference chain in `run_sur.py`. `statsmodels` is an
optional dependency of `run_bayesian_gmm.py` (it fetches the Old Faithful dataset; a built-in
synthetic fallback with matching statistics is used automatically if `statsmodels` isn't
installed).

All scripts insert `src/` onto `sys.path` themselves, so no install step or `PYTHONPATH` setup is
needed — just run them from `experiments/`.

## Reproducing the experiments

Each example in Section 6 has a `run_*.py` script that produces the results and a matching
`plot_*.py` that consumes them. Run scripts default to writing under `experiments/results/`;
plot scripts default to reading from there and writing PDFs under `experiments/plots/`.

| §   | Example                          | Run                          | Plot                                          |
|-----|-----------------------------------|------------------------------|------------------------------------------------|
| 6.1 | Mixture of Gaussians              | `run_mog.py`                 | `plot_mog.py`                                   |
| 6.2 | Mixture of skew-normals           | `run_skew.py`                | `plot_skew.py`                                  |
| 6.3 | Bayesian Gaussian mixture model   | `run_bayesian_gmm.py`        | `plot_bayesian_gmm.py`                          |
| 6.4 | Fetal deaths in litters (T×N sweep) | `run_fetal_deaths.py`      | `plot_fetal_deaths.py --mode lines\|traces`      |
| 6.5 | Sensor network localization       | `run_sensor_network.py`      | `plot_sensor_network.py`                        |
| 6.6 | Seemingly unrelated regression    | `run_sur.py`                 | `plot_sur.py`                                   |

Most scripts are single-shot:

```bash
cd experiments/
python run_mog.py --seed 0                      # → results/gmm_d20_comparison.npz
python plot_mog.py                               # → plots/mog-particle-trace.pdf

python run_sensor_network.py                     # → results/sensor_network.npz
python plot_sensor_network.py

python run_sur.py                                # → results/sur/case2.npz
python plot_sur.py
```

`run_bayesian_gmm.py` and `run_fetal_deaths.py` instead use a two-phase `--mode prepare`/`--mode
run` workflow: `prepare` runs the (seed-independent) annealed-Langevin warm start and trains the
score network once, saving shared state; `run` loads that shared state and produces one
`seed_NNNN.npz` per seed, so the expensive prepare step isn't repeated across seeds.

```bash
python run_bayesian_gmm.py --mode prepare
python run_bayesian_gmm.py --mode run --seed 0   # repeat --seed for more trials
python plot_bayesian_gmm.py                      # violin plot + summary table across seeds

python run_fetal_deaths.py --mode prepare
python run_fetal_deaths.py --mode run --seed 0   # T×N sweep for one seed
python plot_fetal_deaths.py --mode lines         # sensitivity curves across seeds
python plot_fetal_deaths.py --mode traces        # trace plots at 4 representative (T, N)
```

Pass `--help` to any script for its exact options.

## Documentation

- [`docs/targets.md`](docs/targets.md) — the six target distributions and their parameterizations.
- [`docs/nonequilibrium_dds.md`](docs/nonequilibrium_dds.md) — derivation of the nonequilibrium
  DDS controlled SDE used to train MAD-Path's proposal, and why it's better-conditioned than a
  standard (equilibrium) diffusion sampler.

## Citation

The paper is not yet published. A citation (BibTeX) will be added here once it is available.
