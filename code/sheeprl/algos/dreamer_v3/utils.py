from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Sequence

import gymnasium as gym
import numpy as np
import torch
from lightning import Fabric
import torch
from torch import Tensor, nn
from scipy import ndimage

from sheeprl.utils.env import make_env
from sheeprl.utils.imports import _IS_MLFLOW_AVAILABLE
from sheeprl.utils.utils import unwrap_fabric

from torch.distributions.kl import kl_divergence
from torch.distributions import Independent, OneHotCategoricalStraightThrough
from sheeprl.algos.dreamer_v3.cnn_classifier import AnomalyAutoencoder
import os
import cv2
import time
import json
import random

if TYPE_CHECKING:
    from mlflow.models.model import ModelInfo

    from sheeprl.algos.dreamer_v3.agent import PlayerDV3

# ~~~~~~~~~~ EVALUATION CONFIG ~~~~~~~~~~ #
#obstacle regime of this evaluation pass, selects the ground truth operator. must match OBSTACLE_REGIME in
#carla_env_dream.py (which spawns the obstacles). the classifier is the same in both regimes (see CLASSIFIER_PATH)
#  "single" -> one static car on the route (paper R1). ground truth = detect_object(cls=14): a car blob in the
#              camera that the imagination does not reconstruct as a car (this was utils_static_obstacle.py)
#  "random" -> random vehicle / walker / prop on the route (paper R2). ground truth = detect_road_obstacle:
#              class agnostic, any object on the road corridor that the imagination does not reconstruct
OBSTACLE_REGIME = "random"

#safety supervisor mode of this run. one mode per run, the four modes are compared across runs
#the system under test is "both", the others are the reference and the ablations (paper Sec. IV-E)
#  "baseline" -> detectors are computed and logged but never fed back to the car (open loop, as in the paper)
#  "s1"       -> only the cnn classifier controls the car: persistent obstacle flag -> progressive brake to a stop
#  "s2"       -> only the kl novelty rule controls the car: sustained novelty alarm -> S2_ACTION
#  "both"     -> the two tier anomaly observation system: the cnn brakes the car, the kl alarm only gates how much
#                the cnn is trusted (see AnomalyObservationSystem). the kl never touches the throttle here
SAFETY_MODE = "baseline"

#what the kl novelty alarm does to the car in the standalone "s2" mode (ignored by every other mode)
#  "brake"    -> ramp the brake to a stop, same actuator as s1 / both so the collision rates compare detector against detector
#  "cautious" -> scale the throttle by CAUTIOUS_THROTTLE_SCALE while the alarm is sustained, the kl rule as a slow down observer
S2_ACTION = "brake"

#number of evaluation episodes in this run
TEST_EPISODES = 100

#True -> save every camera / imagination frame pair of this run to DATASET_PATH as the classifier dataset,
#labelled normal / anomaly by the ground truth operator. collect in the "random" regime with SAFETY_MODE = "baseline"
#so the supervisor does not alter the driving. False -> nothing is written to disk
BUILD_DATASET = False

#root folder of the sheeprl checkout, every path below is built from it
BASE_PATH = "/home/konstaoi/Desktop/konstaoi/Dreamer/sheeprl"

#the classifier dataset, collected in the random obstacle regime (the single obstacle case is covered by it).
#layout: <folder>/{normal,anomaly}/{camera,imagination}/{episode}_{timestep}.npy
DATASET_PATH = f"{BASE_PATH}/random_distribution_dataset"

#the one classifier, trained by train_classifier.py on DATASET_PATH and used in both evaluation regimes.
#loaded if the file exists, otherwise an untrained network is built (only useful for BUILD_DATASET runs)
CLASSIFIER_PATH = f"{BASE_PATH}/models/classifier/model_checkpoint_2.h5"

#decision threshold on the classifier's sigmoid output: >= threshold -> anomaly flag. value printed by the threshold
#analysis of train_classifier.py (roc / pr on the validation split, largest tau with recall >= 0.81), paper Sec. V-A
CLASSIFIER_THRESHOLD = 0.148534

#a safe stop farther than this from the obstacle counts as a nuisance stop by distance (independent of the ground truth
#label, which fires on reconstruction errors as well), reported next to the label based definition
NUISANCE_STOP_DISTANCE = 25.0

#True -> store a per frame trace (speed, obstacle distance, detector bits, supervisor command) in every episode record,
#needed to plot how the supervisor acted inside an episode. ~10 numbers per frame, a few MB per 100 episode run
SAVE_FRAME_TRACES = True

#safety supervisor parameters (used by "s1", "s2" and "both")
#the brake is progressive: every consecutive frame with the flag up adds S1_BRAKE_RAMP to the brake, up to a full stop.
#isolated flags (reconstruction errors are mostly inconsistent frame to frame) never reach a strong brake, a real
#obstacle stays flagged and does. a gap longer than S1_GAP_TOLERANCE frames resets the ramp and releases the brake
S2_EMA = 0.9                   #exponential moving average factor of the per frame kl novelty bit, 0 = raw bit, closer to 1 = slower, smoother rate
S2_THRESHOLD = 0.5             #the smoothed novelty rate must exceed this for system 2 to raise its alarm (world model unreliable)
S1_BRAKE_RAMP = 0.25           #brake added per consecutive flagged frame (0.25 -> full brake on the 4th consecutive frame, 0.4 s at 10 Hz)
S1_GAP_TOLERANCE = 1           #frames without the flag that are tolerated inside a run. a longer gap resets the ramp to zero
S1_DEBOUNCE_ALARM = 3          #"both": while the s2 alarm is up the ramp starts only after this many consecutive flagged frames
CAUTIOUS_THROTTLE_SCALE = 0.5  #"s2" with S2_ACTION "cautious" only: multiplier applied to the agent's throttle while the alarm is up

AGGREGATOR_KEYS = {
    "Rewards/rew_avg",
    "Game/ep_len_avg",
    "Loss/world_model_loss",
    "Loss/value_loss",
    "Loss/policy_loss",
    "Loss/observation_loss",
    "Loss/reward_loss",
    "Loss/state_loss",
    "Loss/continue_loss",
    "State/kl",
    "State/post_entropy",
    "State/prior_entropy",
    "Grads/world_model",
    "Grads/actor",
    "Grads/critic",
}
MODELS_TO_REGISTER = {"world_model", "actor", "critic", "target_critic", "moments"}

CITYSCAPES_PALLETE = {
    (128, 64, 128):1, # road
    (244, 35, 232):2, # sidewalk
    (70, 70, 70):3, # building
    (102, 102, 156):4, # wall
    (190, 153, 153):5, # fence
    (153, 153, 153):6, # pole
    (250, 170, 30):7, # traffic light
    (220, 220, 0):8, # traffic sign
    (107, 142, 35):9, # vegetation
    (152, 251, 152):10, # terrain
    (70, 130, 180):11, # sky
    (220, 20, 60):12, # pedestrian
    (255, 0, 0):13, # rider
    (0, 0, 142):14, # car
    (0, 0, 56):15, # truck
    (0, 60, 100):16, # bus
    (0, 80, 100):17, # train
    (0, 0, 230):18, # motorcycle
    (119, 11, 32):19 # bicycle
}

CITYSCAPES_CLASS_MAP = ["unallocated", "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign", "vegetation", "terrain", "sky", "pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]

#running low / high percentiles of the imagined returns, used by DreamerV3 to normalize the actor's advantages (sheeprl)
class Moments(nn.Module):
    def __init__(
        self,
        decay: float = 0.99,
        max_: float = 1e8,
        percentile_low: float = 0.05,
        percentile_high: float = 0.95,
    ) -> None:
        super().__init__()
        self._decay = decay
        self._max = torch.tensor(max_)
        self._percentile_low = percentile_low
        self._percentile_high = percentile_high
        self.register_buffer("low", torch.zeros((), dtype=torch.float32))
        self.register_buffer("high", torch.zeros((), dtype=torch.float32))

    def forward(self, x: Tensor, fabric: Fabric) -> Any:
        gathered_x = fabric.all_gather(x).float().detach()
        low = torch.quantile(gathered_x, self._percentile_low)
        high = torch.quantile(gathered_x, self._percentile_high)
        self.low = self._decay * self.low + (1 - self._decay) * low
        self.high = self._decay * self.high + (1 - self._decay) * high
        invscale = torch.max(1 / self._max, self.high - self.low)
        return self.low.detach(), invscale.detach()


#lambda returns of the imagined trajectories for the critic targets (sheeprl)
def compute_lambda_values(
    rewards: Tensor,
    values: Tensor,
    continues: Tensor,
    lmbda: float = 0.95,
):
    vals = [values[-1:]]
    interm = rewards + continues * values * (1 - lmbda)
    for t in reversed(range(len(continues))):
        vals.append(interm[t] + continues[t] * lmbda * vals[-1])
    ret = torch.cat(list(reversed(vals))[:-1])
    return ret


#numpy observation dict -> torch tensors on the fabric device, images are scaled to [-0.5, 0.5] (sheeprl)
def prepare_obs(
    fabric: Fabric, obs: Dict[str, np.ndarray], *, cnn_keys: Sequence[str] = [], num_envs: int = 1, **kwargs
) -> Dict[str, Tensor]:
    torch_obs = {}
    for k, v in obs.items():
        torch_obs[k] = torch.from_numpy(v.copy()).to(fabric.device).float()
        if k in cnn_keys:
            torch_obs[k] = torch_obs[k].view(1, num_envs, -1, *v.shape[-2:]) / 255 - 0.5
        else:
            torch_obs[k] = torch_obs[k].view(1, num_envs, -1)

    return torch_obs

#quantization + restriction operators of the paper (Sec. IV-C): a (1, 1, 3, 64, 64) frame in [-0.5, 0.5] is turned
#into the class index map of the 24x24 drivable corridor crop by nearest colour mapping to the CityScapes palette
def preprocess(x, crop=True):
    #remove batch/time dims
    x = x[0,0]
    #convert from CHW -> HWC
    x = np.transpose(x, (1,2,0))
    
    #take normalized image and turn it to color values   
    x = (x + 0.5) * 255.0
    x = np.clip(x, 0, 255).astype(np.uint8)
    
    #crop to focus erea. set crop=False to keep the FULL frame so small
    #obstacles near the edges aren't clipped out of the focus window.
    if crop:
        x = x[34:58, 20:44]
    
    #conver bgr -> rgb
    x = x[:, :, ::-1]
    
    #map to cityscapes pallete (vectorised nearest-colour; identical result to
    #the per-pixel loop but fast enough to run on the full uncropped frame)
    h, w, _ = x.shape
    colors  = np.array(list(CITYSCAPES_PALLETE.keys()), dtype=np.int32)    # (C,3)
    classes = np.array(list(CITYSCAPES_PALLETE.values()), dtype=np.int32)  # (C,)

    flat = x.reshape(-1, 3).astype(np.int32)                               # (N,3)
    dist = np.sum((flat[:, None, :] - colors[None, :, :]) ** 2, axis=2)    # (N,C)
    class_map = classes[np.argmin(dist, axis=1)].reshape(h, w).astype(np.int32)

    return class_map
    
#normalized class frequency histogram of a class index map (earlier anomaly score component, see test_mode)
def class_histogram(seg, num_classes=20):
    hist = np.zeros(num_classes)
    valid = seg >= 0
    classes, counts = np.unique(seg[valid], return_counts=True)
    hist[classes] = counts
    hist /= hist.sum()
    
    return hist
    
#kl divergence between two class histograms (earlier anomaly score component, see test_mode)
def kl_divergence_custom(p,q):
    eps = 1e-8
    return np.sum(p*np.log((p+eps)/(q+eps)))
    
#fraction of pixels whose class differs between the camera and the imagination crop (earlier anomaly score component)
def pixel_dissagreement(c, i):
    total = (c >=0) & (i >= 0)
    diff = c != i
    
    return diff[total].sum() / total.sum()
    
#log class frequencies of a class index map as a categorical logit vector
def logits(seg, num_classes=19):
    seg = torch.tensor(seg, dtype=torch.long)
    onehot = nn.functional.one_hot(seg, num_classes).float()
    probs = onehot.mean(dim=(0,1))
    logits = torch.log(probs + 1e-8)
    
    return logits
    
#ground truth operator of the single obstacle regime (paper Sec. IV-C, class conditional): number of connected
#blobs of class cls in the camera whose overlap with the same class in the imagination is below the coverage ratio 0.3
def detect_object(cam_seg, img_seg, cls):
    labels, num = ndimage.label(cam_seg == cls)
    missing = 0
    
    for i in range(1, num+1):
        mask = labels == i
        overlap = np.sum(mask & (img_seg == cls))
        ratio = overlap / mask.sum()
        
        if ratio < 0.3:
            missing += 1
    
    return missing

# Classes that make up the normal, static driving scene. Anything NOT in this
# set is treated as a foreign object placed in the scene (a potential obstacle).
BACKGROUND_CLASSES = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}
# 0 unallocated, 1 road, 2 sidewalk, 3 building, 4 wall, 5 fence, 6 pole,
# 7 traffic light, 8 traffic sign, 9 vegetation, 10 terrain, 11 sky

#generalization of detect_object to every non background class: blobs of any foreign class in the camera whose
#same class overlap in the imagination is below the coverage ratio. an intermediate version, superseded by detect_road_obstacle
def detect_missing_obstacles(cam_seg, img_seg, background_classes=BACKGROUND_CLASSES,
                             overlap_threshold=0.3, min_pixels=1, debug=False):
    """Generalises `detect_object` from cls=14 to ANY foreign class.

    Same per-blob overlap logic as `detect_object`: each obstacle blob in the
    camera is flagged when the imagination reconstructs less than
    `overlap_threshold` of it with the SAME class. The ratio is normalised by
    the blob's own size, so a genuinely missed obstacle has ratio ~0 and is
    flagged whether it is large or small.

    NOTE on small obstacles: if a small obstacle is NOT being flagged, it is
    almost always because it never reaches `cam_seg` as a foreign blob (clipped
    by the focus crop in `preprocess`, or blended into 'road' by interpolation
    before the nearest-colour mapping). Run with debug=True to print every
    foreign blob found and its size/ratio so you can see where it is lost.

    Args:
        min_pixels: drop camera blobs smaller than this. Default 1 = keep all.
        debug: print every foreign blob (even sub-threshold ones).

    Returns:
        list of dicts (cls, name, pixels, ratio), one per MISSED obstacle.
    """
    missed = []
    present = np.unique(cam_seg)
    foreign = [int(c) for c in present if int(c) not in background_classes]

    for cls in foreign:
        name = CITYSCAPES_CLASS_MAP[cls] if 0 <= cls < len(CITYSCAPES_CLASS_MAP) else str(cls)
        labels, num = ndimage.label(cam_seg == cls)
        for i in range(1, num + 1):
            mask = labels == i
            size = int(mask.sum())
            overlap = int(np.sum(mask & (img_seg == cls)))
            ratio = overlap / size if size else 0.0
            flagged = (size >= min_pixels) and (ratio < overlap_threshold)

            if debug:
                tag = "MISSED" if flagged else ("too small" if size < min_pixels else "reconstructed")
                print(f"   [debug] {name:<12} blob={size:3d}px overlap={overlap:3d} ratio={ratio:.2f} -> {tag}")

            if flagged:
                missed.append({"cls": cls, "name": name, "pixels": size, "ratio": ratio})

    return missed

# Drivable surface the world model expects to continue ahead. An object that
# appears here in the camera but NOT in the imagination is an obstacle,
# whatever Cityscapes class its colour happens to quantise to.
ROAD_CLASSES = {1, 2, 3, 4, 5, 6, 9}  # 1 road, 2 sidewalk, 5 fence

#ground truth operator of the random obstacle regime (paper Sec. IV-C, class agnostic): connected components of
#non road, non sky classes in the camera, inside the dilated road corridor of both fields, that the imagination
#reconstructs as road / background. labels the classifier dataset offline and the evaluation frames online
def detect_road_obstacle(cam_seg, img_seg, road_classes=ROAD_CLASSES,
                         min_pixels=3, corridor_dilation=2, recon_tolerance=1,
                         debug=False):
    """Class-AGNOSTIC, fully dynamic obstacle detector.

    An obstacle is defined by the RELATION between the two images, not by the
    object's class, so it generalises to anything. A pixel is part of an
    obstacle when ALL three hold:

      (1) ON THE ROAD  - it lies in the drivable corridor, derived dynamically
          as the union of road seen by the camera AND road predicted by the
          imagination, dilated by `corridor_dilation`. Using the imagination's
          road too means a large obstacle that occludes most of the road in the
          camera is still covered (the model still predicts clear road behind
          it). No fixed ROI, no obstacle-class list.

      (2) IT IS A SOLID OBJECT in the camera - not drivable surface, not sky.

      (3) THE IMAGINATION COULDN'T RECREATE IT - the imagination shows clear
          road / background (not an object) at that pixel, within
          `recon_tolerance` to forgive small spatial shifts. This is applied
          per-pixel BEFORE grouping, so normal roadside furniture and the ego
          hood - which the model DOES reconstruct - are removed first and can
          neither be flagged nor bridge a real obstacle into one blob.

    Surviving pixels are grouped into connected blobs; blobs >= `min_pixels`
    are reported with the dominant camera class as the label.

    Returns list of dicts: classes, names, dominant, pixels.
    """
    road_arr = np.array(sorted(road_classes))

    cam_road = np.isin(cam_seg, road_arr)
    img_road = np.isin(img_seg, road_arr)

    # (1) dynamic drivable corridor, from both views
    corridor = ndimage.binary_dilation(cam_road | img_road, iterations=corridor_dilation)

    # (2) solid object in the camera (not drivable surface, not sky)
    cam_object = (~cam_road) & (cam_seg != 11)
    # any object in the imagination (class-agnostic), dilated for tolerance
    img_object = (~img_road) & (img_seg != 11)
    img_object_tol = ndimage.binary_dilation(img_object, iterations=recon_tolerance)

    # (3) novel = camera object the imagination did NOT reconstruct, on the road
    novel = cam_object & (~img_object_tol) & corridor

    labels, num = ndimage.label(novel)
    found = []
    for i in range(1, num + 1):
        mask = labels == i
        size = int(mask.sum())
        if size < min_pixels:
            if debug:
                print(f"   [debug] blob {size}px -> too small")
            continue

        vals, counts = np.unique(cam_seg[mask], return_counts=True)
        cls_list = [int(v) for v in vals]
        names = [CITYSCAPES_CLASS_MAP[v] if 0 <= v < len(CITYSCAPES_CLASS_MAP) else str(v) for v in cls_list]
        dominant = int(vals[np.argmax(counts)])
        dom_name = CITYSCAPES_CLASS_MAP[dominant] if 0 <= dominant < len(CITYSCAPES_CLASS_MAP) else str(dominant)
        found.append({"classes": cls_list, "names": names, "dominant": dom_name, "pixels": size})

        if debug:
            print(f"   [debug] obstacle {size}px classes={names} dominant={dom_name} -> FLAGGED")

    return found
    
#class of the largest camera / imagination disagreement and the fraction of the crop it covers (earlier anomaly score component)
def detect_obstacle(cam_seg, img_seg):
    diff = []
    h, w = cam_seg.shape
    
    #mark differnce classes
    for i in range(h):
        for j in range(w):
            if cam_seg[i,j] != img_seg[i,j]:
                diff.append(cam_seg[i,j])
                
    obstacle = [0 ,0]

    #if differnce class takes up more than 5% of image flag as obstacle
    for obs in list(diff):
        perc = list(diff).count(obs)/(h * w)
        if perc > 0.03 and perc > obstacle[1]:
            obstacle = [obs, perc]

    return obstacle

#confusion matrix and the prevalence robust metrics of the paper (balanced accuracy, mcc, paper eq. 13) of one detector
def evaluate_predictions(ground_truth, predictions, method_name="Method", verbose=True):
    """
    Compute a full set of classification metrics suited to imbalanced
    binary anomaly detection.

    Args:
        ground_truth: array-like of labels. Can be 0/1 or counts
            (from `detect_object`); values > 0 are treated as "anomaly".
        predictions: array-like of 0/1 predictions.
        method_name: label used in the printout.
        verbose: if True, print a formatted report.

    Returns:
        dict with confusion matrix entries and derived metrics.
    """
    gt = np.asarray(ground_truth).astype(int)
    pr = np.asarray(predictions).astype(int)

    # Binarize — detect_object returns a COUNT, not 0/1
    gt = (gt > 0).astype(int)
    pr = (pr > 0).astype(int)

    # Confusion matrix
    TP = int(np.sum((pr == 1) & (gt == 1)))
    TN = int(np.sum((pr == 0) & (gt == 0)))
    FP = int(np.sum((pr == 1) & (gt == 0)))
    FN = int(np.sum((pr == 0) & (gt == 1)))

    total = TP + TN + FP + FN
    n_pos = TP + FN  # real anomalies
    n_neg = TN + FP  # real normal frames
    eps = 1e-12

    accuracy = (TP + TN) / max(total, 1)
    precision = TP / max(TP + FP, eps)                 # of flagged, how many real
    recall = TP / max(TP + FN, eps)                    # TPR — fraction of anomalies caught
    specificity = TN / max(TN + FP, eps)               # TNR — fraction of normals left alone
    fpr = FP / max(FP + TN, eps)                       # false alarm rate
    fnr = FN / max(FN + TP, eps)                       # miss rate (critical for safety)

    f1 = 2 * precision * recall / max(precision + recall, eps)
    balanced_acc = (recall + specificity) / 2          # fixes "always 0" problem

    # Matthews correlation coefficient — robust single-number summary.
    # Returns 0 for any degenerate classifier (always-0, always-1, random).
    mcc_denom = np.sqrt(float((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))
    mcc = (TP * TN - FP * FN) / mcc_denom if mcc_denom > 0 else 0.0

    results = {
        "method": method_name,
        "n_samples": total,
        "n_anomalies": n_pos,
        "anomaly_rate": n_pos / max(total, 1),
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "fpr": fpr,
        "fnr": fnr,
        "f1": f1,
        "mcc": mcc,
    }

    if verbose:
        print(f"\n=== {method_name} ===")
        print(f"Samples: {total}  "
              f"(anomalies: {n_pos} [{100 * n_pos / max(total, 1):.1f}%], "
              f"normal: {n_neg})")
        print("\nConfusion matrix:")
        print("               pred=0     pred=1")
        print(f"   gt=0    TN={TN:6d}   FP={FP:6d}")
        print(f"   gt=1    FN={FN:6d}   TP={TP:6d}")
        print()
        print(f"  Accuracy            : {accuracy:.4f}   (can mislead on imbalanced data)")
        print(f"  Balanced accuracy   : {balanced_acc:.4f}   (0.5 = trivial classifier)")
        print(f"  Precision           : {precision:.4f}   (of flags, fraction that are real)")
        print(f"  Recall / TPR        : {recall:.4f}   (of anomalies, fraction caught)")
        print(f"  Specificity / TNR   : {specificity:.4f}   (of normals, fraction passed)")
        print(f"  False-positive rate : {fpr:.4f}   (false alarms)")
        print(f"  False-negative rate : {fnr:.4f}   (MISSES — safety critical)")
        print(f"  F1 score            : {f1:.4f}")
        print(f"  MCC                 : {mcc:.4f}   (range -1..1, 0 = chance)")

    return results


#side by side table of the detector metrics (the columns of paper tables II - III)
def compare_methods(results_list):
    """Pretty-print a side-by-side comparison table across methods."""
    rows = [
        ("Accuracy",            "accuracy"),
        ("Balanced accuracy",   "balanced_accuracy"),
        ("Precision",           "precision"),
        ("Recall (TPR)",        "recall"),
        ("Specificity (TNR)",   "specificity"),
        ("FPR (false alarm)",   "fpr"),
        ("FNR (miss rate)",     "fnr"),
        ("F1",                  "f1"),
        ("MCC",                 "mcc"),
    ]

    header = f"{'Metric':<22} " + "  ".join(
        f"{r['method']:>14}" for r in results_list
    )
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for label, key in rows:
        vals = "  ".join(f"{r[key]:>14.4f}" for r in results_list)
        print(f"{label:<22} {vals}")
    print("=" * len(header))



# ~~~~~~~~~~ ANOMALY OBSERVATION SYSTEM ~~~~~~~~~~ #
#closed loop safety supervisor, turns the two per frame detector bits into a control command (paper Sec. VI-B)
#  baseline -> never intervenes, open loop evaluation
#  s1       -> consecutive cnn flags ramp the brake by S1_BRAKE_RAMP per frame up to a full stop, a gap longer than
#              S1_GAP_TOLERANCE frames resets the ramp and releases the brake
#  s2       -> smoothed kl novelty rate above S2_THRESHOLD raises the s2 alarm, consecutive alarm frames ramp the
#              brake the same way (S2_ACTION "brake") or scale the throttle (S2_ACTION "cautious")
#  both     -> the cnn ramps the brake as in s1, the s2 alarm only gates the trust in the cnn:
#              alarm down -> the ramp starts on the first flag
#              alarm up   -> the ramp starts after S1_DEBOUNCE_ALARM consecutive flags
#the command computed from frame t is applied by the env on the action of frame t+1
SAFETY_MODES = ["baseline", "s1", "s2", "both"]

class AnomalyObservationSystem:
    #one supervisor per run, the mode is fixed for the run
    def __init__(self, mode):
        assert mode in SAFETY_MODES, f"unknown safety mode {mode}, expected one of {SAFETY_MODES}"
        self.mode = mode
        self.reset()

    #clears the temporal state at the start of every episode
    def reset(self):
        self.s2_rate = 0.0 #smoothed novelty rate
        self.s1_run = 0    #consecutive frames with the cnn flag up (gaps up to S1_GAP_TOLERANCE do not break the run)
        self.s1_gap = 0    #consecutive frames without the cnn flag since the run started
        self.s2_run = 0    #consecutive frames with the s2 alarm up (standalone s2 only)

    #advances a run counter with the gap tolerance: the run grows on a flag, survives short gaps, resets on long ones
    def _advance(self, run, gap, flag):
        if flag:
            return run + 1, 0
        if run > 0 and gap < S1_GAP_TOLERANCE:
            return run, gap + 1
        return 0, 0

    #one frame: cnn bit and kl bit in, control command out
    def update(self, s1_flag, s2_flag):
        s1_flag = s1_flag > 0
        s2_flag = s2_flag > 0

        #system 2: sustained distribution shift -> alarm (the world model is unreliable)
        self.s2_rate = S2_EMA * self.s2_rate + (1 - S2_EMA) * float(s2_flag)
        s2_alarm = self.mode in ["s2", "both"] and self.s2_rate > S2_THRESHOLD

        #system 1: persistence of the obstacle flag, the alarm delays the start of the ramp
        self.s1_run, self.s1_gap = self._advance(self.s1_run, self.s1_gap, s1_flag)
        s1_trusted = self.mode == "s1" or (self.mode == "both" and not s2_alarm)
        delay = 0 if s1_trusted else S1_DEBOUNCE_ALARM

        brake = 0.0
        if self.mode in ["s1", "both"]:
            brake = min(1.0, max(0, self.s1_run - delay) * S1_BRAKE_RAMP)

        #standalone system 2: the alarm itself drives the car
        throttle_scale = 1.0
        if self.mode == "s2":
            self.s2_run = self.s2_run + 1 if s2_alarm else 0
            if S2_ACTION == "brake":
                brake = min(1.0, self.s2_run * S1_BRAKE_RAMP)
            elif s2_alarm:
                throttle_scale = CAUTIOUS_THROTTLE_SCALE

        return {"brake": brake, "throttle_scale": throttle_scale, "s2_alarm": s2_alarm, "s1_trusted": s1_trusted, "s2_rate": self.s2_rate, "s1_run": self.s1_run}

#map the env terminal flags to one safety outcome
def episode_outcome(info, last_brake):
    if bool(info.get("collision", False)):
        return "collision"
    if bool(info.get("lane_invasion", False)):
        return "lane_departure"
    if bool(info.get("safe_stop", False)):
        return "safe_stop"
    if info.get("destination_reward", 0.0) > 0:
        return "route_complete"
    if bool(info.get("time_exceeded", False)):
        return "safe_stop" if last_brake > 0 else "timeout"
    return "waypoint_lost"

SAFETY_OUTCOMES = ["collision", "safe_stop", "lane_departure", "route_complete", "waypoint_lost", "timeout"]

#columns of the per frame trace stored in every episode record when SAVE_FRAME_TRACES is True
TRACE_FIELDS = ["step", "speed", "obstacle_distance", "gt", "s1", "s2", "s2_rate", "brake", "throttle_scale", "s2_alarm"]

#aggregate the per episode safety records of one supervisor mode
def summarize_safety(episodes, mode, verbose=True):
    n = len(episodes)
    counts = {o: sum(1 for e in episodes if e["outcome"] == o) for o in SAFETY_OUTCOMES}

    non_collision = [e for e in episodes if e["outcome"] != "collision" and not np.isnan(e["min_obstacle_distance"])]
    latencies = [e["first_brake"] - e["gt_onset"] for e in episodes if e["gt_onset"] is not None and e["first_brake"] is not None]
    impact_speeds = [e["impact_speed"] for e in episodes if e["impact_speed"] is not None]
    stop_margins = [e["stop_margin"] for e in episodes if e["stop_margin"] is not None]
    first_brake_distances = [e["distance_at_first_brake"] for e in episodes if e["distance_at_first_brake"] is not None]
    total_frames = sum(e["steps"] for e in episodes)

    summary = {
        "mode": mode,
        "episodes": n,
        "episodes_with_gt_obstacle": sum(1 for e in episodes if e["gt_onset"] is not None),
        **{f"{o}_rate": counts[o] / max(n, 1) for o in SAFETY_OUTCOMES},
        "mean_route_completion": float(np.mean([e["route_completion"] for e in episodes])) if n else 0.0,
        "mean_min_obstacle_distance": float(np.mean([e["min_obstacle_distance"] for e in non_collision])) if non_collision else float("nan"),
        "mean_intervention_latency": float(np.mean(latencies)) if latencies else float("nan"),
        "mean_distance_at_first_brake": float(np.mean(first_brake_distances)) if first_brake_distances else float("nan"),
        "mean_impact_speed": float(np.mean(impact_speeds)) if impact_speeds else float("nan"),
        "mean_stop_margin": float(np.mean(stop_margins)) if stop_margins else float("nan"),
        "median_stop_margin": float(np.median(stop_margins)) if stop_margins else float("nan"),
        "nuisance_stop_rate": sum(1 for e in episodes if e["nuisance_stop"]) / max(n, 1),
        "nuisance_stop_distance_rate": sum(1 for e in episodes if e.get("nuisance_stop_distance")) / max(n, 1),
        "outcomes_by_obstacle_type": {t: {o: sum(1 for e in episodes if e.get("obstacle_type") == t and e["outcome"] == o) for o in SAFETY_OUTCOMES if any(e.get("obstacle_type") == t and e["outcome"] == o for e in episodes)} for t in sorted({e.get("obstacle_type") for e in episodes if e.get("obstacle_type")})},
        "brake_frame_fraction": sum(e["brake_frames"] for e in episodes) / max(total_frames, 1),
        "full_brake_frame_fraction": sum(e["full_brake_frames"] for e in episodes) / max(total_frames, 1),
        "nuisance_brake_fraction": sum(e["nuisance_brake_frames"] for e in episodes) / max(total_frames, 1),
        "s2_alarm_frame_fraction": sum(e["s2_alarm_frames"] for e in episodes) / max(total_frames, 1),
        "mean_episode_reward": float(np.mean([e["reward"] for e in episodes])) if n else 0.0,
        "mean_episode_steps": float(np.mean([e["steps"] for e in episodes])) if n else 0.0,
    }

    if verbose:
        print(f"\n=== Safety [{mode}] ===")
        print(f"Episodes: {n}")
        for o in SAFETY_OUTCOMES:
            print(f"  {o:<15}: {counts[o]:4d}  ({100 * counts[o] / max(n, 1):5.1f}%)")
        print(f"  Route completion      : {summary['mean_route_completion']:.3f}")
        print(f"  Min obstacle distance : {summary['mean_min_obstacle_distance']:.2f} m   (non collision episodes)")
        print(f"  Intervention latency  : {summary['mean_intervention_latency']:.2f} steps after GT onset, {summary['mean_distance_at_first_brake']:.2f} m from the obstacle")
        print(f"  Impact speed          : {summary['mean_impact_speed']:.2f} m/s   (collision episodes)")
        print(f"  Stop margin           : {summary['mean_stop_margin']:.2f} m mean, {summary['median_stop_margin']:.2f} m median   (safe stop episodes)")
        print(f"  Nuisance stops        : {100 * summary['nuisance_stop_rate']:.1f}% by label (stop before any GT obstacle), {100 * summary['nuisance_stop_distance_rate']:.1f}% by distance (stop > {NUISANCE_STOP_DISTANCE:.0f} m from the obstacle)")
        if summary["outcomes_by_obstacle_type"]:
            print("  Outcomes by obstacle type:")
            for t, outs in summary["outcomes_by_obstacle_type"].items():
                print(f"    {t:<40s} " + "  ".join(f"{o}={c}" for o, c in outs.items()))
        print(f"  Brake frames          : {100 * summary['brake_frame_fraction']:.1f}%   (full brake: {100 * summary['full_brake_frame_fraction']:.1f}%, nuisance, GT=0: {100 * summary['nuisance_brake_fraction']:.1f}%)")
        print(f"  S2 alarm frames       : {100 * summary['s2_alarm_frame_fraction']:.1f}%")
        print(f"  Mean reward / steps   : {summary['mean_episode_reward']:.1f} / {summary['mean_episode_steps']:.0f}")

    return summary

#load the summaries saved by separate runs (safety_eval_<mode>.json) to compare them with compare_safety()
def load_safety_summaries(paths):
    summaries = []
    for path in paths:
        with open(path) as f:
            summaries.append(json.load(f)["summary"])
    return summaries

#side by side table of the safety summaries of the supervisor modes, e.g.
#compare_safety(load_safety_summaries(["<run>/safety_eval_baseline.json", "<run>/safety_eval_s1.json", ...]))
def compare_safety(summaries):
    rows = [
        ("Collision rate",        "collision_rate"),
        ("Safe stop rate",        "safe_stop_rate"),
        ("Lane departure rate",   "lane_departure_rate"),
        ("Route complete rate",   "route_complete_rate"),
        ("Waypoint lost rate",    "waypoint_lost_rate"),
        ("Route completion",      "mean_route_completion"),
        ("Min obstacle dist (m)", "mean_min_obstacle_distance"),
        ("Latency (steps)",       "mean_intervention_latency"),
        ("First brake dist (m)",  "mean_distance_at_first_brake"),
        ("Impact speed (m/s)",    "mean_impact_speed"),
        ("Stop margin (m)",       "mean_stop_margin"),
        ("Nuisance stop rate",    "nuisance_stop_rate"),
        ("Nuisance stop (dist.)",  "nuisance_stop_distance_rate"),
        ("Brake frames",          "brake_frame_fraction"),
        ("Full brake frames",     "full_brake_frame_fraction"),
        ("Nuisance brake frames", "nuisance_brake_fraction"),
        ("S2 alarm frames",       "s2_alarm_frame_fraction"),
        ("Mean reward",           "mean_episode_reward"),
    ]

    header = f"{'Metric':<22} " + "  ".join(f"{s['mode']:>10}" for s in summaries)
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for label, key in rows:
        vals = "  ".join(f"{s[key]:>10.4f}" for s in summaries)
        print(f"{label:<22} {vals}")
    print("=" * len(header))


#entry point of the evaluation pass (called by dreamer_v3.py after training with cfg.algo.run_test):
#builds the env and the classifier of the selected regime and runs TEST_EPISODES episodes under SAFETY_MODE
@torch.no_grad()
def test(
    player: "PlayerDV3",
    fabric: Fabric,
    cfg: Dict[str, Any],
    log_dir: str,
    test_name: str = "",
    greedy: bool = True,
):
    """Test the model on the environment with the frozen model.

    Args:
        player (PlayerDV3): the agent which contains all the models needed to play.
        fabric (Fabric): the fabric instance.
        cfg (DictConfig): the hyper-parameters.
        log_dir (str): the logging directory.
        test_name (str): the name of the test.
            Default to "".
        greedy (bool): whether or not to sample the actions.
            Default to True.
    """
    torch.cuda.empty_cache()
    env: gym.Env = make_env(cfg, cfg.seed, 0, log_dir, "test" + (f"_{test_name}" if test_name != "" else ""))()
    
    s = env.unwrapped.client.get_world().get_settings()
    print("sync:", s.synchronous_mode, "dt:", s.fixed_delta_seconds)
    
    classifier = AnomalyAutoencoder(CLASSIFIER_PATH, CLASSIFIER_THRESHOLD)
    
    if BUILD_DATASET:
        for folder in ["normal", "anomaly"]:
            for kind in ["camera", "imagination"]:
                os.makedirs(f"{DATASET_PATH}/{folder}/{kind}", exist_ok=True)
    
    #seed the route / obstacle draws so the four supervisor modes (separate runs) face the same episodes
    if cfg.seed is None:
        print("WARNING: cfg.seed is None, the episodes of this run are not paired with the other supervisor modes")
    random.seed(cfg.seed)
    summary = test_mode(player, fabric, cfg, log_dir, env, classifier, SAFETY_MODE, greedy)
    
    env.close()
    return summary

#one full evaluation pass under a single supervisor mode (paper Sec. IV-E and VI). every step: the frozen agent
#acts, both detectors are computed on the same frame (system 1 on the paired class maps, system 2 on the rssm
#divergences), the ground truth operator labels the frame, the supervisor turns the bits into the command of the
#next step, and the detection metrics and the safety record of the episode are updated
@torch.no_grad()
def test_mode(player, fabric, cfg, log_dir, env, classifier, safety_mode, greedy):
    #done = False
    cumulative_rew = 0
    obs = env.reset(seed=cfg.seed)[0]
    player.num_envs = 1
    player.init_states()
    
    supervisor = AnomalyObservationSystem(safety_mode)
    safety_episodes = [] #one record per episode, saved to safety_eval_<mode>.json
    print(f"\n##### Safety supervisor mode: {safety_mode} #####")
    
    training_eps = TEST_EPISODES
    episode = 0
    total_steps = 0
    
    #taken from 97.5th percentile
    pixel_threshold = 0.1 #% 0.3055 #grid 0.4306
    kl_threshold = 0.2 #% 0.2973 #grid 0.6492
    anomaly_threshold = 0.1 #% 0.2467 #grid 0.4711
    
    pixel_batch = [] 
    kl_batch = []  
    anomaly_batch = [] 
    
    #metrics 
    ground_truth_metric = []
    classifier_metric = []
    novelty_metric = []
    brake_engaged_metric = [] #1 when the brake override was active while the frame was processed
    
    while episode < training_eps:
        print("Episode:", episode)
        policy_step = 0
        done = False
        
        env.set_current_episode(episode)   
        supervisor.reset()
        env.set_safety_command(0.0, 1.0)
        
        #safety record of this episode
        ep = {
            "episode": episode, "mode": safety_mode, "reward": 0.0,
            "gt_onset": None, "first_brake": None, "first_s1": None, "first_s2": None,
            "brake_frames": 0, "full_brake_frames": 0, "nuisance_brake_frames": 0, "s2_alarm_frames": 0,
            "s1_frames": 0, "s2_frames": 0, "gt_frames": 0,
            "min_obstacle_distance": float("nan"),
            "distance_at_gt_onset": None,    #ego -> obstacle distance when the ground truth first flags it
            "distance_at_first_brake": None, #ego -> obstacle distance when the brake first engages
            "impact_speed": None,            #speed on the frame before the collision (collision episodes)
            "stop_margin": None,             #ego -> obstacle distance at the safe stop (safe stop episodes)
            "nuisance_stop": False,          #safe stop triggered before the ground truth ever saw the obstacle
            "nuisance_stop_distance": False, #safe stop farther than NUISANCE_STOP_DISTANCE from the obstacle
            "obstacle_type": None,           #carla blueprint of the injected obstacle
            "trace": [],                     #per frame, see TRACE_FIELDS
        }
        last_info = {}
        last_cmd = {"brake": 0.0, "throttle_scale": 1.0, "s2_alarm": False, "s2_rate": 0.0}
        prev_speed = 0.0

        while not done:
            # Act greedly through the environment       
            torch_obs = prepare_obs(fabric, obs, cnn_keys=cfg.algo.cnn_keys.encoder)
            mask = {k: v for k, v in torch_obs.items() if k.startswith("mask")}
            actions_raw = player.get_actions(torch_obs, greedy, mask)
        
            real_actions = actions_raw[0]
            actions = actions_raw
            #actions = torch.cat(actions_raw, -1).cpu().numpy()
            #step_data["actions"] = actions.reshape((1, cfg.env.num_envs, -1))
        
            if player.actor.is_continuous:
                real_actions = torch.stack(real_actions, -1).cpu().numpy()
            else:
                real_actions = torch.stack([real_act.argmax(dim=-1) for real_act in real_actions], dim=-1).cpu().numpy()

            # Single environment step
            obs, reward, done, truncated, info = env.step(real_actions.reshape(env.action_space.shape))
            done = done or truncated or cfg.dry_run
            cumulative_rew += reward
            ep["reward"] += float(reward)
            prev_speed = float(last_info.get("speed_norm", 0.0))
            last_info = info
            
            obstacle_distance = info.get("obstacle_distance", float("nan"))
            if not np.isnan(obstacle_distance):
                ep["min_obstacle_distance"] = np.nanmin([ep["min_obstacle_distance"], obstacle_distance])
        
            #test to get logits                
            prior_logits, posterior_logits, reconstructed_obs, is_novelty = player.get_states(cfg, actions[0], policy_step, torch_obs)
            
            camera_obs = torch_obs["semantic_camera"].cpu()
            camera_img = reconstructed_obs["semantic_camera"].cpu() 

            env.set_observation(camera_obs.numpy())
            env.set_imagination(camera_img.numpy())
            
            co = preprocess(camera_obs.numpy())
            ci = preprocess(camera_img.numpy())
            
            #metric #1 class based pixel dissagreement
            pixel_dissagreement_error = pixel_dissagreement(co, ci)
            #if pixel_dissagreement_error > pixel_threshold:
            #    print("Pixel Anomaly!!")
            #print("Pixles dissagreement:",  pixel_dissagreement_error)
            
            #metrc #2 KL Divergence
            cam_hist = class_histogram(co)
            img_hist = class_histogram(ci)
            
            #distribution_error = kl_divergence_custom(cam_hist, img_hist)
            #if distribution_error > kl_threshold:
            #    print("KL Anomaly!!")
            #print("KL divergence:", distribution_error)
            
            #detect obstacle and percentage
            #obstacle = detect_obstacle(co, ci)
            #print(f"Class {obstacle[0]} {obstacle[1]} %")
            
            #anomaly score
            #score = distribution_error*0.3 + pixel_dissagreement_error*0.4 + obstacle[1]*0.3
            #if score > anomaly_threshold: 
            #    print("Anomaly Score Anomaly!!")
            #print("Anomaly Score:", score)
            
            #metric #3 object detection
            
            #predict with classifier
            concat_img = np.stack((co, ci), axis=-1)
            concat_img = np.expand_dims(concat_img, axis=0)
            
            anomaly_detection = classifier.predict(concat_img)
            env.set_anomaly_detection(anomaly_detection)
            #if anomaly_detection > 0.1: 
            #    print(f"Object {CITYSCAPES_CLASS_MAP[obstacle[0]]}: {obstacle[1]*100:.3f} %")
                #env.set_anomaly_detection(True)
            
            #ground truth
            if OBSTACLE_REGIME == "single":
                #single obstacle regime: a car (class 14) present in the camera but not in the imagination
                missing_object = detect_object(co, ci, cls=14)
            else:
                #random obstacle regime, class agnostic: any object on the road the imagination didn't predict,
                #whatever class its colour quantises to (e.g. a prop mapped to wall / pole). debug=True prints every blob
                obstacles = detect_road_obstacle(co, ci)
                missing_object = len(obstacles)
                #if obstacles:
                #    detected = ", ".join(f"{o['dominant']} ({o['pixels']}px)" for o in obstacles)
                #    print(f"Obstacle(s) on road not imagined: {detected}")
            #print("object", missing_object)
            #if pixel_dissagreement_error > pixel_threshold and distribution_error > kl_threshold and score > anomaly_threshold and missing_object < 1:
            #    print("Danger!")
            #if pixel_dissagreement_error > pixel_threshold or distribution_error > kl_threshold or score > anomaly_threshold and missing_object < 1:
            #    print("Causion")
            #if missing_object > 0:
            #    print("Anomaly!!!")
            #    print(f"Class {obstacle[0]} {obstacle[1]*100:.2f} %")
            #if missing_object > 0:
            #    print("Truth Anomaly!!!")
            #print("Truth:", missing_object)
            
            #if missing_object and score > threshold:
            #   print(f"Class {obstacle[0]} {obstacle[1]*100:.3f} %")

            #build dataset
            if BUILD_DATASET:
                folder = "anomaly" if missing_object > 0 else "normal"
                np.save(f"{DATASET_PATH}/{folder}/camera/{episode}_{policy_step}.npy", camera_obs)
                np.save(f"{DATASET_PATH}/{folder}/imagination/{episode}_{policy_step}.npy", camera_img)
        
            #pixel_batch.append(pixel_dissagreement_error)
            #kl_batch.append(distribution_error)
            #anomaly_batch.append(score)
            
            #print("GT:", missing_object)
            #print("Class", anomaly_detection)
            #print("Novel", is_novelty)
            
            ground_truth_metric.append(missing_object)
            classifier_metric.append(anomaly_detection)
            novelty_metric.append(is_novelty)
            brake_engaged_metric.append(int(last_cmd["brake"] > 0))
            
            #closed loop: detector bits -> supervisor -> env command for the next step
            cmd = supervisor.update(anomaly_detection, is_novelty)
            env.set_safety_command(cmd["brake"], cmd["throttle_scale"])
            last_cmd = cmd
            
            #safety record
            gt = missing_object > 0
            ep["gt_frames"] += int(gt)
            ep["s1_frames"] += int(anomaly_detection > 0)
            ep["s2_frames"] += int(is_novelty > 0)
            ep["brake_frames"] += int(cmd["brake"] > 0)
            ep["full_brake_frames"] += int(cmd["brake"] >= 1.0)
            ep["nuisance_brake_frames"] += int(cmd["brake"] > 0 and not gt)
            ep["s2_alarm_frames"] += int(cmd["s2_alarm"])
            if gt and ep["gt_onset"] is None:
                ep["gt_onset"] = policy_step
            if anomaly_detection > 0 and ep["first_s1"] is None:
                ep["first_s1"] = policy_step
            if is_novelty > 0 and ep["first_s2"] is None:
                ep["first_s2"] = policy_step
            if cmd["brake"] > 0 and ep["first_brake"] is None:
                ep["first_brake"] = policy_step
                ep["distance_at_first_brake"] = None if np.isnan(obstacle_distance) else float(obstacle_distance)
            if gt and ep["distance_at_gt_onset"] is None:
                ep["distance_at_gt_onset"] = None if np.isnan(obstacle_distance) else float(obstacle_distance)
            
            if SAVE_FRAME_TRACES:
                ep["trace"].append([policy_step, float(info.get("speed_norm", 0.0)), float(obstacle_distance), int(gt),
                                    int(anomaly_detection > 0), int(is_novelty > 0), round(cmd["s2_rate"], 4),
                                    cmd["brake"], cmd["throttle_scale"], int(cmd["s2_alarm"])])
            
            policy_step += 1
        
        if cfg.metric.log_level > 0 and len(fabric.loggers) > 0:
            fabric.logger.log_metrics({"Test/cumulative_reward": cumulative_rew}, 0)
        
        #close the safety record of this episode
        completed = last_info.get("num_completed", 0)
        obsolete = last_info.get("num_obsolete", 0)
        ep["steps"] = policy_step
        ep["outcome"] = episode_outcome(last_info, last_cmd["brake"])
        ep["route_completion"] = completed / max(completed + obsolete, 1)
        ep["final_speed"] = float(last_info.get("speed_norm", float("nan")))
        ep["final_obstacle_distance"] = float(last_info.get("obstacle_distance", float("nan")))
        ep["obstacle_type"] = last_info.get("obstacle_type", None)
        if ep["outcome"] == "collision":
            ep["impact_speed"] = prev_speed
        if ep["outcome"] == "safe_stop":
            ep["stop_margin"] = None if np.isnan(ep["final_obstacle_distance"]) else ep["final_obstacle_distance"]
            ep["nuisance_stop"] = ep["gt_onset"] is None or (ep["first_brake"] is not None and ep["first_brake"] < ep["gt_onset"])
            ep["nuisance_stop_distance"] = ep["stop_margin"] is not None and ep["stop_margin"] > NUISANCE_STOP_DISTANCE
        safety_episodes.append(ep)
        print(f"Outcome: {ep['outcome']}  min obstacle distance: {ep['min_obstacle_distance']:.2f} m  brake frames: {ep['brake_frames']}  s2 alarm frames: {ep['s2_alarm_frames']}")
        
        #new episode: fresh observation and a cleared recurrent state, so the kl monitor's history is this episode's only
        obs = env.reset()[0]
        player.init_states()
        episode += 1
        total_steps += policy_step
        
        print(f"~~~ Episode {episode} Results ~~~")
        print("Steps:", total_steps)
        classifier_results = evaluate_predictions(ground_truth_metric, classifier_metric, "Classifier")
        novelty_results = evaluate_predictions(ground_truth_metric, novelty_metric, "KL Novelty")
        print("")        
        compare_methods([classifier_results, novelty_results])
        
        #pixel_values = torch.tensor(pixel_batch)
        #pixel_threshold = torch.quantile(pixel_values, 0.975)
        #print("Pixel 97.5th percentile threshold:", pixel_threshold.item())
        
        #kl_values = torch.tensor(kl_batch)
        #kl_threshold = torch.quantile(kl_values, 0.975)
        #print("KL 97.5th percentile threshold:", kl_threshold.item())
        
        #anomaly_values = torch.tensor(anomaly_batch)
        #anomaly_threshold = torch.quantile(anomaly_values, 0.975)
        #print("Anomaly 97.5th percentile threshold:", anomaly_threshold.item())
        
        #save every 10 episodes
        #if episode % 10 == 0 :
            #autoencoder.save(total_steps)
    
    #summarize the mode and save everything next to the sheeprl logs
    summary = summarize_safety(safety_episodes, safety_mode)
    #detection metrics on all frames, and on the frames processed before the brake engaged: once the car is
    #stopped the world model's reconstruction is no longer a valid reference, so those frames are reported apart
    gt_all = np.array(ground_truth_metric)
    free = np.array(brake_engaged_metric) == 0
    summary["detection"] = {
        "classifier": evaluate_predictions(gt_all, classifier_metric, "Classifier", verbose=False),
        "kl_novelty": evaluate_predictions(gt_all, novelty_metric, "KL Novelty", verbose=False),
    }
    summary["detection_pre_intervention"] = {
        "classifier": evaluate_predictions(gt_all[free], np.array(classifier_metric)[free], "Classifier (pre intervention)", verbose=False),
        "kl_novelty": evaluate_predictions(gt_all[free], np.array(novelty_metric)[free], "KL Novelty (pre intervention)", verbose=False),
    }
    
    #everything needed to reproduce the run and read the episode records later
    summary["config"] = {
        "obstacle_regime": OBSTACLE_REGIME, "safety_mode": safety_mode, "s2_action": S2_ACTION, "test_episodes": TEST_EPISODES, "seed": cfg.seed,
        "s2_ema": S2_EMA, "s2_threshold": S2_THRESHOLD, "s1_brake_ramp": S1_BRAKE_RAMP, "s1_gap_tolerance": S1_GAP_TOLERANCE,
        "s1_debounce_alarm": S1_DEBOUNCE_ALARM, "cautious_throttle_scale": CAUTIOUS_THROTTLE_SCALE, "classifier": CLASSIFIER_PATH, "classifier_threshold": CLASSIFIER_THRESHOLD,
    }
    
    out_path = f"{log_dir}/safety_eval_{safety_mode}.json"
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "trace_fields": TRACE_FIELDS, "episodes": safety_episodes}, f, indent=2, default=float)
    print("Safety evaluation saved to", out_path)
    
    return summary

#weight initialization of DreamerV3 (Hafner et al.), truncated normal scaled by the fan in / fan out (sheeprl)
# Adapted from: https://github.com/NM512/dreamerv3-torch/blob/main/tools.py#L929
def init_weights(m):
    if isinstance(m, nn.Linear):
        in_num = m.in_features
        out_num = m.out_features
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(m.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        space = m.kernel_size[0] * m.kernel_size[1]
        in_num = space * m.in_channels
        out_num = space * m.out_channels
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(m.weight.data, mean=0.0, std=std, a=-2.0, b=2.0)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.LayerNorm):
        m.weight.data.fill_(1.0)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


#uniform weight initialization with a given scale, used for the output layers (sheeprl)
# Adapted from: https://github.com/NM512/dreamerv3-torch/blob/main/tools.py#L957
def uniform_init_weights(given_scale):
    def f(m):
        if isinstance(m, nn.Linear):
            in_num = m.in_features
            out_num = m.out_features
            denoms = (in_num + out_num) / 2.0
            scale = given_scale / denoms
            limit = np.sqrt(3 * scale)
            nn.init.uniform_(m.weight.data, a=-limit, b=limit)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)
        elif isinstance(m, nn.LayerNorm):
            m.weight.data.fill_(1.0)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)

    return f


#registers the world model, actor and critic of a checkpoint with mlflow (sheeprl)
def log_models_from_checkpoint(
    fabric: Fabric, env: gym.Env | gym.Wrapper, cfg: Dict[str, Any], state: Dict[str, Any]
) -> Sequence["ModelInfo"]:
    if not _IS_MLFLOW_AVAILABLE:
        raise ModuleNotFoundError(str(_IS_MLFLOW_AVAILABLE))
    import mlflow  # noqa

    from sheeprl.algos.dreamer_v3.agent import build_agent

    # Create the models
    is_continuous = isinstance(env.action_space, gym.spaces.Box)
    is_multidiscrete = isinstance(env.action_space, gym.spaces.MultiDiscrete)
    actions_dim = tuple(
        env.action_space.shape
        if is_continuous
        else (env.action_space.nvec.tolist() if is_multidiscrete else [env.action_space.n])
    )
    world_model, actor, critic, target_critic = build_agent(
        fabric,
        actions_dim,
        is_continuous,
        cfg,
        env.observation_space,
        state["world_model"],
        state["actor"],
        state["critic"],
        state["target_critic"],
    )
    moments = Moments(
        fabric,
        cfg.algo.actor.moments.decay,
        cfg.algo.actor.moments.max,
        cfg.algo.actor.moments.percentile.low,
        cfg.algo.actor.moments.percentile.high,
    )
    moments.load_state_dict(state["moments"])

    # Log the model, create a new run if `cfg.run_id` is None.
    model_info = {}
    with mlflow.start_run(run_id=cfg.run.id, experiment_id=cfg.experiment.id, run_name=cfg.run.name, nested=True) as _:
        model_info["world_model"] = mlflow.pytorch.log_model(unwrap_fabric(world_model), artifact_path="world_model")
        model_info["actor"] = mlflow.pytorch.log_model(unwrap_fabric(actor), artifact_path="actor")
        model_info["critic"] = mlflow.pytorch.log_model(unwrap_fabric(critic), artifact_path="critic")
        model_info["target_critic"] = mlflow.pytorch.log_model(target_critic, artifact_path="target_critic")
        model_info["moments"] = mlflow.pytorch.log_model(moments, artifact_path="moments")
        mlflow.log_dict(cfg.to_log, "config.json")
    return model_info
