# Installation and reproduction manual

This document gets a fresh Linux machine from zero to running every experiment of the
paper/thesis. Operating the system once installed (run matrix, global variables,
outputs) is covered by the companion `MANUAL.md`; this file covers installation and
the order of reproduction.

The distribution is an **overlay**, not a fork: you install CARLA and SheepRL from
their own upstream sources, then drop our files on top. Nothing else in SheepRL is
modified.

---

## 0. What you need

| Requirement | Used in our runs |
|---|---|
| OS | Ubuntu Linux (20.04/22.04 class), x86-64 |
| GPU | NVIDIA GPU with a CUDA 12 driver (we ran on driver CUDA 12.7). CARLA and DreamerV3 share it. |
| Python | **3.8.10** (the pinned package set is for 3.8; newer interpreters will not resolve it) |
| Disk | ~25 GB: CARLA package ~17 GB, venv ~8 GB, plus room for logs/checkpoints/dataset |
| RAM | 32 GB recommended (CARLA + DreamerV3 replay buffer) |

Three upstream components, all pinned:

1. **CARLA 0.9.15**, prebuilt release package (no source build required).
2. **SheepRL**, git clone, installed editable (`pip install -e .`).
3. **Python packages** exactly as in the provided `requirements.txt`
   (`torch==2.4.1`, `tensorflow==2.8.0`, `gymnasium==0.29.1`, `carla==0.9.15`, …).
   PyTorch (DreamerV3) and TensorFlow (the CNN classifier) coexist in the same venv;
   the pins in `requirements.txt` are a combination known to work — in particular
   `protobuf==3.20.0` and `numpy==1.24.3`, which both frameworks tolerate.

## 1. Contents of the distribution

Two archives. `aos_code.zip` (this one) mirrors the target tree:

```
INSTALL.md                  this file
MANUAL.md                   how to run: globals, run matrix A–F, outputs
requirements.txt            exact pip freeze of the working venv
code/
  PythonAPI/                vendored package (edited CARLA agents/navigation code +
                            the CarDreamer toolkit planner); lives at the REPO ROOT,
                            next to sheeprl.py — not inside the simulator package.
                            Only agents/ ships: the CARLA client itself comes from
                            the pip package, not from eggs or wheels in this folder
  sheeprl/
    cli.py                                  modified CLI entry
    utils/env.py                            env factory, knows the id `carla_env_dream`
    envs/carla_env_dream.py                 the CARLA Gymnasium environment (the copy
                                            the factory imports; its globals select
                                            scenario and obstacle regime)
    algos/dreamer_v3/
      agent.py                              PlayerDV3 with the KL `get_states` method
      utils.py                              detectors, ground-truth operators, safety
                                            supervisor, eval loop
      cnn_classifier.py                     the CNN binary classifier (Keras)
      train_classifier.py                   classifier training / threshold selection
    configs/algo/dreamer_v3.yaml
    configs/env/carla_env_dream.yaml        referenced as `env=carla_env_dream`
    configs/exp/dreamer_v3.yaml             the experiment config (hyperparameters)
```

`aos_models.zip` contains the trained models, so retraining is optional:

```
models/
  dreamer_v3/
    s1_static_route_ckpt_131000_0.ckpt      scenario S1 agent
    s2_random_route_ckpt_246000_0.ckpt      scenario S2 agent
    s3_random_towns_ckpt_535000_0.ckpt      scenario S3 agent
  classifier/
    model_checkpoint_2.h5                   the trained CNN classifier
```

**Not included**(*): the collected classifier dataset and our raw result JSONs;
available from the authors on request.

## 2. CARLA 0.9.15

1. Download the prebuilt **CARLA 0.9.15** Linux release archive from the official
   CARLA GitHub releases and extract it, e.g. to `/opt/carla-simulator`. The package
   already contains the server binary (`CarlaUE4.sh`) and the `PythonAPI/` folder
   with examples; nothing is installed system-wide.
2. Start the server (own terminal, leave it running):

   ```
   /opt/carla-simulator/CarlaUE4.sh -RenderOffScreen -nosound -carla-server
   ```

   `-RenderOffScreen` runs headless; drop it if you want the spectator window. The
   server listens on the default port 2000, which is what the environment expects.
3. The Python *client* is **not** taken from the package's egg files: it is the pip
   package `carla==0.9.15` inside the venv (installed in the next step). Client and
   server versions must match exactly — both are 0.9.15 here.

The standard towns used by the experiments (Town01 and the rotation list of the
random-towns scenario) ship inside the base package; no additional maps package is
needed.

## 3. Python environment

```
python3.8 -m venv ~/venvs/aos          # must be Python 3.8
source ~/venvs/aos/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` is a verbatim freeze of the working machine. It contains one line
you must **delete before installing**:

```
sheeprl @ file:///home/.../sheeprl
```

(that line records our local editable install; SheepRL is installed in the next step
instead).

GPU notes: `torch==2.4.1` ships its own CUDA 12.1 runtime wheels, which run fine on a
CUDA 12.x driver. `tensorflow==2.8.0` is only used by the classifier (a tiny network);
it may run CPU-only without affecting any result or timing that matters.

## 4. SheepRL

```
git clone https://github.com/Eclectic-Sheep/sheeprl.git
cd sheeprl
git checkout <SHEEPRL_COMMIT>          # the commit recorded in our repo (*)
pip install -e . --no-deps
```

`--no-deps` keeps the pinned versions from `requirements.txt` authoritative. The
clone provides the two entry points used below: `sheeprl.py` (training) and
`sheeprl_eval.py` (evaluation).

(*) Pin the commit you verify against. On the original machine it is printed by
`git -C <sheeprl checkout> rev-parse HEAD`.

## 5. Apply the overlay

From the distribution root, with `<repo>` being your SheepRL clone:

```
cp -r code/PythonAPI <repo>/                 # vendored package at the repo root
cp -r code/sheeprl   <repo>/                 # merges into the package tree
```

The second copy **replaces** five upstream files (`cli.py`, `utils/env.py`,
`algos/dreamer_v3/agent.py`, `algos/dreamer_v3/utils.py`,
`configs/algo/dreamer_v3.yaml` and `configs/exp/dreamer_v3.yaml`) and adds the rest.
Replacing them is intended: each contains the upstream code plus our additions, at
the pinned commit of Section 4. Two cleanups if you are upgrading an older copy of
the overlay: delete `algos/dreamer_v3/utils_static_obstacle.py` and any stale
`algos/dreamer_v3/carla_env_dream.py` — the live environment is the one at
`sheeprl/envs/carla_env_dream.py`, which `utils/env.py` imports, and a leftover copy
under `algos/` would be the wrong file to edit. `dreamer_v3.py` itself is used
unmodified. Note that the exp config selects the `dreamer_v3_S` model size
(`override /algo: dreamer_v3_S`); the hyperparameters of the paper/thesis (batch 16,
sequence 64, buffer $10^5$, $5\times10^6$ total steps, `semantic_camera`/`state`
keys) come from `configs/exp/dreamer_v3.yaml`.

## 6. Expected folder layout at runtime

All commands below run from the root of the SheepRL checkout, inside the venv, with
the CARLA server already running.

```
<repo>/
  PythonAPI/                                       # from the overlay
  logs/runs/dreamer_v3/carla_env_dream/<run>/version_0/checkpoint/ckpt_*.ckpt
  models/classifier/model_checkpoint_2.h5          # from aos_models.zip, or produced by train_classifier.py
  models/dreamer_v3/*.ckpt                         # from aos_models.zip (optional)
  random_distribution_dataset/                     # produced by a BUILD_DATASET run (*)
    normal/  anomalous/
```

Commands must run **from the repo root**, so that the vendored `PythonAPI` package is
importable. `models/classifier/model_checkpoint_2.h5` and
`random_distribution_dataset/` are the values of `CLASSIFIER_PATH` and `DATASET_PATH`
in `utils.py` / `train_classifier.py`; keep them consistent (see MANUAL.md §2).

## 7. Reproduction, in order

**Step 1 — train the three agents** (skip it if you use the checkpoints from
`aos_models.zip`). One run per driving scenario; set `ROUTE_SCENARIO` in
`sheeprl/envs/carla_env_dream.py` (`"static"`, `"random"`, `"random_towns"`) and
`OBSTACLE_REGIME = "none"`, then:

```
python3 sheeprl.py exp=dreamer_v3 env=carla_env_dream fabric.accelerator=cuda checkpoint.every=1000
```

Training can be resumed from the latest checkpoint at any time by appending
`checkpoint.resume_from="./logs/runs/dreamer_v3/carla_env_dream/<run>/version_0/checkpoint/ckpt_<N>_0.ckpt"`.
Our runs trained to roughly 130k (static), 300k (random route) and 550k (random
towns) environment steps; the shipped checkpoints are the final ones of those runs
(131k, 246k and 535k respectively).

**Step 2 — collect the classifier dataset.** With the *random-route* checkpoint:
`ROUTE_SCENARIO="random"`, `OBSTACLE_REGIME="random"` in **both** files,
`SAFETY_MODE="baseline"`, `BUILD_DATASET=True` in `utils.py`, then run an evaluation
pass (this is the frozen-agent command used for every evaluation below):

```
python3 sheeprl_eval.py +env=carla_env_dream checkpoint_path="<ckpt of step 1>" fabric.accelerator=cuda env.capture_video=False
```

100 episodes populate `random_distribution_dataset/`. Set `BUILD_DATASET=False`
afterwards. (*) Our dataset is available on request.

**Step 3 — train the classifier and select its threshold.**

```
python3 ./sheeprl/algos/dreamer_v3/train_classifier.py
```

with `TRAIN_CLASSIFIER=True`: episode-level 70/15/15 split, training, then the
ROC-based threshold selection on the validation episodes (criterion: largest τ with
recall ≥ 0.81). It writes `models/classifier/model_checkpoint_2.h5`,
`threshold_analysis.png/.csv`, and prints the selected τ — paste it into
`CLASSIFIER_THRESHOLD` in `utils.py` (ours: `0.148534`). Steps 2 and 3 are skipped
entirely when using the shipped `model_checkpoint_2.h5`, whose threshold is already
the default in `utils.py`.

**Step 4 — the 24 evaluation runs.** For each of the 6 conditions (2 obstacle
regimes × 3 scenarios) run the four supervisor configurations
(`SAFETY_MODE = "baseline" | "s2" | "s1" | "both"`), 100 episodes each, using the
`sheeprl_eval.py` command of step 2 with the matching scenario checkpoint. The full
matrix, the globals to set per run, and the output JSON schema are in MANUAL.md
(runs A–F). Remember the one consistency rule: `OBSTACLE_REGIME` must hold the same
value in `carla_env_dream.py` and `utils.py`.

Episodes are drawn independently on purpose (no fixed seed): configurations are
compared through the 95% Wilson intervals of their rates, as in the paper/thesis.
Expect rates to match the published tables within those intervals, not digit for
digit.

## 8. Smoke test (15 minutes)

After installation, before any long run:

1. CARLA server up; venv active; `python -c "import carla, torch, tensorflow"` exits
   clean.
2. Set `ROUTE_SCENARIO="static"` and `OBSTACLE_REGIME="single"` in
   `sheeprl/envs/carla_env_dream.py`, `OBSTACLE_REGIME="single"`,
   `SAFETY_MODE="baseline"`, `TEST_EPISODES=3` in `utils.py`.
3. Run the `sheeprl_eval.py` command with the S1 checkpoint from `aos_models.zip`
   (`checkpoint_path="models/dreamer_v3/s1_static_route_ckpt_131000_0.ckpt"`).
   Success looks like: the agent drives the static route, most of the 3 episodes end
   in a collision with the parked car (this is the unsupervised baseline), and a
   results JSON appears containing `summary.config` with exactly the globals you set
   and per-episode `trace` arrays. Switching `SAFETY_MODE` to `"s1"` and re-running
   should then end the episodes in safe stops ~18 m before the obstacle.

## 9. Troubleshooting

- **Client/server version mismatch error** — the pip `carla` version must be 0.9.15,
  same as the server package.
- **CARLA exits immediately headless** — the machine needs Vulkan-capable NVIDIA
  drivers; `-RenderOffScreen` still renders on the GPU.
- **TF 2.8 import errors about protobuf** — reinstall `protobuf==3.20.0` (something
  upgraded it).
- **Hydra: unknown env `carla_env_dream`** — `configs/env/carla_env_dream.yaml` or
  the modified `utils/env.py` was not copied in.
- **`ModuleNotFoundError: PythonAPI`** — the command was not run from the repo root,
  or the vendored `PythonAPI/` folder was not copied next to `sheeprl.py`.
- **Evaluation hangs at episode start** — the CARLA server was started after the
  client, or on a non-default port; restart the server first, then the run.
