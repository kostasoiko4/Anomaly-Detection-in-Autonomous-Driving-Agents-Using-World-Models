# Running the Anomaly Observation System: manual

This manual covers the four files in `anomaly_observation_system_code.zip` and every run needed for the paper. Everything is selected through the global variables at the top of each file; nothing is commented in or out any more.

---

## 1. File map

The code files live in your SheepRL checkout as laid out in INSTALL.md §1: `utils.py`, `agent.py`, `cnn_classifier.py` and `train_classifier.py` under `sheeprl/algos/dreamer_v3/`, and `carla_env_dream.py` under `sheeprl/envs/` (this is the copy the environment factory imports; delete any stale copy under `algos/dreamer_v3/`). `utils_static_obstacle.py` is superseded by `utils.py` and can be deleted.

| File | What it is | Globals you set |
|---|---|---|
| `carla_env_dream.py` | The CARLA environment. Used both by DreamerV3 training and by the evaluation passes. | `ROUTE_SCENARIO`, `OBSTACLE_REGIME`, static coordinates, town list, `SAFE_STOP_*` |
| `utils.py` | Detectors, ground truth operators, the safety supervisor, the evaluation loop `test()` / `test_mode()`, metrics. | `OBSTACLE_REGIME`, `SAFETY_MODE`, `S2_ACTION`, `TEST_EPISODES`, `BUILD_DATASET`, `SAVE_FRAME_TRACES`, `DATASET_PATH`, `CLASSIFIER_PATH`, supervisor parameters |
| `cnn_classifier.py` | The System 1 CNN classifier (Keras). Takes the checkpoint path as an argument, no globals. | none |
| `train_classifier.py` | Classifier training / evaluation on the collected dataset, episode-level split. | `TRAIN_CLASSIFIER`, `SPLIT_BY`, `SPLIT_FRACTIONS`, `DATASET_PATH`, `CLASSIFIER_PATH` |

`agent.py`, `dreamer_v3.py` and the yaml files are unchanged.

### The one consistency rule

`OBSTACLE_REGIME` exists in two files and **must hold the same value** in both for any evaluation run:

- `carla_env_dream.py` decides what is spawned on the road,
- `utils.py` decides which ground truth operator labels the frames.

Values: `"single"` (paper R1, one static car), `"random"` (paper R2, random vehicle / walker / prop). The environment additionally accepts `"none"`, used only when training the agent. There is **one classifier**, trained on the random-obstacle dataset and used in both regimes; `CLASSIFIER_PATH` must be the same string in `utils.py` and `train_classifier.py`.

---

## 2. Global variables, what each one does

### `carla_env_dream.py`

```
ROUTE_SCENARIO   "static" | "random" | "random_towns"     paper S1 | S2 | S3
OBSTACLE_REGIME  "none" | "single" | "random"             training | R1 | R2
```

- `"static"` loads `STATIC_TOWN`, every episode starts at `STATIC_SPAWN_LOCATION` / `STATIC_SPAWN_YAW` and drives from `STATIC_INITIAL_LOCATION` to `STATIC_DESTINATION`.
- `"random"` loads `RANDOM_TOWN`, random spawn point, route of `ROUTE_LENGTH_M` meters, resampled until it has `MIN_ROUTE_WAYPOINTS` waypoints.
- `"random_towns"` loads a random town from `TOWN_LIST` and swaps it every `TOWN_CHANGE_EVERY` episodes.
- `SINGLE_OBSTACLE_BLUEPRINT` is the CARLA blueprint of the R1 obstacle; `RANDOM_OBSTACLE_FILTERS` is the pool of the R2 regime (vehicles and static props; walkers are excluded because CARLA registers no collision when the ego vehicle pushes a pedestrian). `OBSTACLE_MIN_WAYPOINT` keeps the obstacle away from the start and the end of the route, and `OBSTACLE_STRAIGHT_LOOKBACK` / `OBSTACLE_MAX_APPROACH_TURN_DEG` place it on a straight approach (heading change over the preceding waypoints below the limit), so the obstacle is in view before the car reaches it. `OBSTACLE_PLACEMENT_TRIES` random indices are tried before the constraint is dropped for that episode.
- `SAFE_STOP_TERMINATION` / `SAFE_STOP_SPEED` / `SAFE_STOP_STEPS`: evaluation only. Once the supervisor has the brake engaged and the car is below `SAFE_STOP_SPEED` for `SAFE_STOP_STEPS` consecutive steps, the episode is truncated. It never triggers during training since nothing engages the brake there.

### `utils.py`

```
OBSTACLE_REGIME     "single" | "random"                 must match the env
SAFETY_MODE         "baseline" | "s1" | "s2" | "both"   one per run
S2_ACTION           "brake" | "cautious"                only for SAFETY_MODE = "s2"
TEST_EPISODES       100                                 episodes of the run
BUILD_DATASET       False | True                        write the classifier dataset during the run
SAVE_FRAME_TRACES   True | False                        per frame trace in every episode record
```

- `baseline`: detectors are computed and logged, the car is never touched. This is the open-loop pass of Tables I-III and the reference of Table V.
- `s1`: the CNN alone brakes the car.
- `s2`: the KL alarm alone acts on the car, with `S2_ACTION` = `"brake"` (same ramp as the others) or `"cautious"` (throttle scaled).
- `both`: the two tier system. The CNN brakes; while the KL alarm is up the ramp starts only after `S1_DEBOUNCE_ALARM` consecutive flags. The KL never touches the throttle here.

**The brake is progressive.** Every consecutive frame with the flag up adds `S1_BRAKE_RAMP` to the brake (0.25 -> full brake on the 4th frame). A gap of up to `S1_GAP_TOLERANCE` frames without the flag keeps the run and holds the brake level; a longer gap resets the ramp to zero and releases the brake. Isolated flags from reconstruction errors therefore produce at most a light brake, a persistent obstacle produces a stop.

Supervisor parameters: `S2_EMA`, `S2_THRESHOLD`, `S1_BRAKE_RAMP`, `S1_GAP_TOLERANCE`, `S1_DEBOUNCE_ALARM`, `CAUTIOUS_THROTTLE_SCALE`. All are written into the JSON of every run under `summary.config`.

Paths: `BASE_PATH`, `DATASET_PATH` (the random-obstacle dataset), `CLASSIFIER_PATH` (the one checkpoint). `CLASSIFIER_THRESHOLD` is the sigmoid decision threshold of the classifier, taken from the threshold analysis of Run C.

### `train_classifier.py`

```
TRAIN_CLASSIFIER  False | True             True trains and saves, False only evaluates the saved checkpoint
SPLIT_BY          "frames" | "episodes"    what the 70/15/15 refers to (always whole episodes)
SPLIT_FRACTIONS   (0.70, 0.15, 0.15)
DATASET_PATH / CLASSIFIER_PATH             same values as in utils.py
```

The number of episodes is read from the file names; `98_465.npy` as the last frame means episodes 0-98 and the split adapts.

---

## 3. The runs, in order

The evaluation entry point is `utils.test()`, called by `dreamer_v3.py` when `cfg.algo.run_test` is set, on the frozen checkpoint of the agent. Use the same command you use today to load a checkpoint and run the test; everything below only changes globals.

Every evaluation run prints `sync: True dt: 0.1` at the start. If it prints `False`, stop, the simulator is not in synchronous mode.

### Run A. Train the agents (one per scenario, already done)

Set in `carla_env_dream.py`:

| Agent | `ROUTE_SCENARIO` | `OBSTACLE_REGIME` |
|---|---|---|
| S1 | `"static"` | `"none"` |
| S2 | `"random"` | `"none"` |
| S3 | `"random_towns"` | `"none"` |

`OBSTACLE_REGIME = "none"` is what guarantees the world model only ever sees obstacle free scenes. Keep one checkpoint per scenario.

### Run B. Collect the classifier dataset (one run)

Set:

| file | global | value |
|---|---|---|
| `carla_env_dream.py` | `ROUTE_SCENARIO` | the scenario of the agent you use for collection |
| `carla_env_dream.py` | `OBSTACLE_REGIME` | `"random"` (the single obstacle case is covered by it) |
| `utils.py` | `OBSTACLE_REGIME` | `"random"` |
| `utils.py` | `SAFETY_MODE` | `"baseline"` (the supervisor must never alter the driving while collecting) |
| `utils.py` | `BUILD_DATASET` | `True` |
| `utils.py` | `TEST_EPISODES` | `100` |

Output: `DATASET_PATH/{normal,anomaly}/{camera,imagination}/{episode}_{timestep}.npy`. The label folder comes from the class-agnostic ground truth operator. The run also writes a `safety_eval_baseline.json`, ignore it for this step.

Set `BUILD_DATASET` back to `False` afterwards. It writes ~50 KB per frame per modality.

### Run C. Train the classifier (one run)

In `train_classifier.py` set `TRAIN_CLASSIFIER = True`, `SPLIT_BY = "frames"`, then:

```
python train_classifier.py
```

Console output to record for the paper (Section V-A):

```
Episodes found: 99 (ids 0-98)  frames: 27636  split by: frames
  train      episodes   0- 69 ( 70)  frames  19345 (70.0%)  anomalous  ...
  validation episodes  70- 84 ( 15)  frames   4148 (15.0%)  anomalous  ...
  test       episodes  85- 98 ( 14)  frames   4143 (15.0%)  anomalous  ...
Test Score: <BCE>   Test Accuracy: <acc>
```

The checkpoint goes to `CLASSIFIER_PATH`. Set `TRAIN_CLASSIFIER = False` to re-evaluate a saved checkpoint without retraining.

**Threshold analysis** (`THRESHOLD_ANALYSIS = True`, default). After the evaluation the script scores the **validation** split, prints ROC-AUC, PR-AUC, the share of scores in the intermediate zone, and a precision / recall / FPR table at `CANDIDATE_THRESHOLDS` plus the selected value, then reports the test split once at that value. Selection criterion is `THRESHOLD_CRITERION`: `"recall_at_least"` (default, largest threshold keeping validation recall >= `MIN_RECALL`), `"youden"` or `"f1"`. The curves go to `THRESHOLD_PLOT_PATH`, the full table to `THRESHOLD_TABLE_PATH`. Copy the printed value into `CLASSIFIER_THRESHOLD` in `utils.py`; it is recorded in every run's JSON under `summary.config.classifier_threshold`.

### Run D. Open-loop detection pass, Tables I-III (six runs)

| file | global | value |
|---|---|---|
| `carla_env_dream.py` | `ROUTE_SCENARIO` | `"static"` / `"random"` / `"random_towns"`, with the matching agent checkpoint |
| `carla_env_dream.py` | `OBSTACLE_REGIME` | `"single"` / `"random"` |
| `utils.py` | `OBSTACLE_REGIME` | same as the env |
| `utils.py` | `SAFETY_MODE` | `"baseline"` |
| `utils.py` | `BUILD_DATASET` | `False` |

3 scenarios x 2 regimes = 6 runs. Each run prints the running detector table every episode (the last print is the final one) and writes `safety_eval_baseline.json`. The detection metrics of Tables II-III are `summary.detection.classifier` and `summary.detection.kl_novelty`; the sample counts of Table I are `n_samples` and `anomaly_rate` inside them.

### Run E. Closed-loop safety pass, Table V (one run per mode)

Same env settings as Run D. In `utils.py` set `SAFETY_MODE` to one value and run; repeat for each mode.

| `SAFETY_MODE` | role in Table V |
|---|---|
| `"baseline"` | reference (identical to the Run D file of the same condition, reuse it) |
| `"s2"` | ablation, system without tier 1 (`S2_ACTION = "brake"`) |
| `"s1"` | ablation, system without tier 2 |
| `"both"` | the system under test |

Primary grid: R1 x {S1, S2, S3} x {s2, s1, both} = 9 runs (baseline comes from Run D). R2 is secondary, same grid.

**Set `cfg.seed` to a fixed integer** for every run of one condition (`seed=None` prints a warning). `test()` reseeds `random` with it so the four runs draw the same routes, obstacle types and obstacle positions; without it the modes are unpaired and cannot be compared episode by episode.

The output file name only carries the mode, so move each JSON into a folder that carries the rest:

```
results/
  R1/S1/safety_eval_baseline.json   safety_eval_s2.json   safety_eval_s1.json   safety_eval_both.json
  R1/S2/...
  R1/S3/...
  R2/...
```

### Run F. Optional parameter sweep (Figure 5 of the evaluation plan)

Repeat Run E with `SAFETY_MODE = "both"` in S3 for a few values of `S1_BRAKE_RAMP` (e.g. 0.2, 0.25, 0.5), `S1_GAP_TOLERANCE` (0, 1, 2) and `S2_THRESHOLD` (e.g. 0.3, 0.5, 0.7). Every JSON records its parameters under `summary.config`, so the files can be told apart after the fact; rename them anyway.

---

## 4. Reading the results

### Console

Every run ends with the safety summary of its mode and the detector tables. To compare the five modes of one condition afterwards, from a Python shell inside the sheeprl environment:

```python
from sheeprl.algos.dreamer_v3.utils import load_safety_summaries, compare_safety
compare_safety(load_safety_summaries([
    "results/R1/S3/safety_eval_baseline.json",
    "results/R1/S3/safety_eval_s2.json",
    "results/R1/S3/safety_eval_s1.json",
    "results/R1/S3/safety_eval_both.json",
]))
```

### The JSON

```
summary
  mode, episodes, episodes_with_gt_obstacle
  collision_rate, safe_stop_rate, lane_departure_rate, route_complete_rate, waypoint_lost_rate, timeout_rate
  mean_route_completion
  mean_min_obstacle_distance          non collision episodes
  mean_intervention_latency           steps, first brake minus GT onset
  mean_distance_at_first_brake        m
  mean_impact_speed                   m/s, collision episodes, speed on the frame before impact
  mean_stop_margin, median_stop_margin   m, safe stop episodes
  nuisance_stop_rate                  safe stops before the GT ever saw the obstacle
  brake_frame_fraction, full_brake_frame_fraction, nuisance_brake_fraction, s2_alarm_frame_fraction
  mean_episode_reward, mean_episode_steps
  detection                    classifier / kl_novelty metrics on all frames (Tables II-III)
  detection_pre_intervention   same, on frames processed before the brake engaged
  config                       regime, mode, s2_action, episodes, seed, the supervisor parameters, the classifier path
trace_fields                   column names of episodes[i].trace
episodes[]                     one record per episode (all fields above at episode level, plus outcome,
                               gt_onset, first_brake, first_s1, first_s2, steps, reward, trace)
```

`trace` is a list of `[step, speed, obstacle_distance, gt, s1, s2, s2_rate, brake, throttle_scale, s2_alarm]` per frame, for the mechanism figure. Set `SAVE_FRAME_TRACES = False` to drop it.

`outcome` per episode is one of `collision`, `safe_stop`, `lane_departure`, `route_complete`, `waypoint_lost`, `timeout` (`timeout` only appears when `SAFE_STOP_TERMINATION` is off and the car idles to the horizon).

---

## 5. Sanity checks worth doing once

1. **Baseline invariance.** In a `baseline` run, `summary.detection` and `summary.detection_pre_intervention` are identical, and `brake_frame_fraction` is 0. If either fails, the supervisor leaked into the run.
2. **`both` equals `s1` in S1 and S2.** The KL alarm stays down there, so the two modes should produce the same episode outcomes up to CARLA nondeterminism. A difference means the alarm is firing where it should be silent.
3. **`s2` in S3 stops almost every episode early** with a high `nuisance_stop_rate`. That is expected; it is the closed-loop form of the KL rule tracking scene familiarity.
4. **Brake level in the trace.** In `s1` / `both` runs the `brake` column of the trace should show ramps (0.25, 0.5, 0.75, 1.0) that collapse to 0 after a two-frame gap; a trace that jumps straight to 1.0 means `S1_BRAKE_RAMP` is set to 1.
5. **Warm-up of the alarm.** With `S2_EMA = 0.9` the alarm needs about 7 consecutive KL frames to cross 0.5 from a cold start. CNN false positives in those first frames of an S3 episode still brake `both`.

---

## 6. Run count

| Step | Runs | Episodes |
|---|---|---|
| B. dataset collection | 1 | 100 |
| C. classifier training | 1 | offline |
| D. open-loop detection | 6 | 600 |
| E. closed-loop safety, R1 | 9 | 900 |
| E. closed-loop safety, R2 (optional) | 9 | 900 |
| F. sweep (optional) | as many as points | 100 each |

---

## 7. Episode boundaries and town changes

Both issues flagged in the first review are fixed in this version: `test_mode()` now takes the observation returned by `env.reset()` and calls `player.init_states()` at every episode start, so the recurrent state (and therefore the KL monitor's history) belongs to the current episode only; and `reset()` re-applies the synchronous 10 Hz world settings and refreshes the spectator and vehicle blueprint after every `load_world()` in `random_towns`. Runs made before this version carried the recurrent state across episodes; their System 2 numbers are not comparable with runs made after it.

## 8. Reading the closed-loop results

- Restrict the collision comparison to episodes where the obstacle actually blocks the lane: with paired seeds, use the episodes in which `baseline` ended in `collision` and report, for each mode, how many of those it avoided. Episodes the baseline passes without contact (small props off the lane centre) say nothing about the supervisor.
- `nuisance_stop_rate` (label based) and `nuisance_stop_distance_rate` (stop farther than `NUISANCE_STOP_DISTANCE` from the obstacle) are both reported; the second is independent of the ground truth label, which also fires on reconstruction discrepancies of the scene.
- `outcomes_by_obstacle_type` in the summary splits the outcomes by CARLA blueprint, to see which obstacle classes the corridor detector catches and at what distance.
