#trains and evaluates the system 1 cnn classifier on the dataset collected by utils.test_mode (BUILD_DATASET),
#split by whole episodes into train / validation / test (paper Sec. V-A)
import cv2
import sys
import os
import random
import tensorflow as tf
import numpy as np
from cnn_classifier import AnomalyAutoencoder
import torch
from torch.distributions import Independent, OneHotCategoricalStraightThrough
from torch.distributions.kl import kl_divergence
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support, auc, confusion_matrix, classification_report
from skimage.metrics import structural_similarity

from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

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

# ~~~~~~~~~~ DATASET CONFIG ~~~~~~~~~~ #
#root folder of the sheeprl checkout, every path below is built from it
base_path = "/home/konstaoi/Desktop/konstaoi/Dreamer/sheeprl"

#the classifier dataset, collected in the random obstacle regime (utils.py BUILD_DATASET). the single obstacle
#case is covered by it, one classifier serves both evaluation regimes. layout:
#  <folder>/normal/camera/{episode}_{timestep}.npy    <folder>/normal/imagination/{episode}_{timestep}.npy
#  <folder>/anomaly/camera/{episode}_{timestep}.npy   <folder>/anomaly/imagination/{episode}_{timestep}.npy
#the episode id is read from the file name, so the number of episodes is whatever the folder contains
DATASET_PATH = f"{base_path}/random_distribution_dataset"

#classifier checkpoint, must match CLASSIFIER_PATH in utils.py. loaded when it exists, overwritten when TRAIN_CLASSIFIER is True
CLASSIFIER_PATH = f"{base_path}/models/classifier/model_checkpoint_2.h5"

#True  -> train a new classifier on the train split (validation split monitored) and save it to CLASSIFIER_PATH
#False -> only load the saved classifier and evaluate it on the test split
TRAIN_CLASSIFIER = False

#True -> roc / pr analysis of the sigmoid output on the VALIDATION split and selection of the decision threshold.
#the test split is scored once at the selected threshold and is never used to pick it. prints the table below,
#saves the curves to THRESHOLD_PLOT_PATH and the full table to THRESHOLD_TABLE_PATH
THRESHOLD_ANALYSIS = True

#how the threshold is selected on the validation split
#  "recall_at_least" -> the LARGEST threshold whose recall is still >= MIN_RECALL (safety conservative: keep the misses
#                       near zero, then remove as many false alarms as that allows). this is the paper's direction
#  "youden"          -> maximum of recall - fpr (the roc point farthest from the diagonal)
#  "f1"              -> maximum f1 on the validation split
THRESHOLD_CRITERION = "recall_at_least"
MIN_RECALL = 0.81   #recall level at which the validation roc flattens (paper Sec. V-A)

#thresholds listed in the printed table, the selected one is added to them
CANDIDATE_THRESHOLDS = [0.000001, 0.0001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9]

THRESHOLD_PLOT_PATH = f"{base_path}/models/classifier/threshold_analysis.png"
THRESHOLD_TABLE_PATH = f"{base_path}/models/classifier/threshold_analysis.csv"

#True -> non-learned baselines for system 1 (paper Sec. V-A, "Non-learned baseline"): the fraction of pixels of the
#24x24 corridor crop where the real and imagined class maps disagree, thresholded with the same rule as the cnn
#(THRESHOLD_CRITERION "recall_at_least", MIN_RECALL) on the validation split and scored once on the test split.
#answers whether a plain disagreement count is enough or the learned classifier is needed
MISMATCH_BASELINE = True
MISMATCH_RESULTS_PATH = f"{base_path}/models/classifier/mismatch_baseline_results.json"

#train / validation / test fractions. the split is ALWAYS done on whole episodes, an episode never lands in two splits
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)

#what the fractions refer to (episodes have different lengths, so the two are not the same)
#  "episodes" -> the sorted episode ids are cut by NUMBER OF EPISODES, the frame share of each split then depends on the episode lengths
#  "frames"   -> episodes are added in id order until each split holds its share of the TOTAL FRAMES, so the frame split is 70/15/15 up to one episode
SPLIT_BY = "frames"

#initialize paths
normal_data_folder = f"{DATASET_PATH}/normal"
anomalous_data_folder = f"{DATASET_PATH}/anomaly"

#peek at the first frame pair to check the array layout
first_frame = sorted(f for f in os.listdir(f"{normal_data_folder}/camera") if f.endswith(".npy"))[0]
obs = np.load(f"{normal_data_folder}/camera/{first_frame}", allow_pickle=True)
img = np.load(f"{normal_data_folder}/imagination/{first_frame}", allow_pickle=True)

print("Dataset:", DATASET_PATH)
print("Obs shape:", obs.shape , "Img shape:", img.shape)
#load normal data set
normal_data = []
#episode id -> list of [cam, img, label] records. Filenames are {episode}_{timestep}.npy,
#so the episode is the part before the first underscore.
episode_frames = {}

camera_folder = f"{normal_data_folder}/camera"
imagination_folder = f"{normal_data_folder}/imagination"
dots = 1

for image in os.listdir(camera_folder):
    if not image.endswith(".npy"):
        continue
    cam_data = np.load(f"{camera_folder}/{image}", allow_pickle=True)
    img_data = np.load(f"{imagination_folder}/{image}", allow_pickle=True)
    record = [cam_data, img_data, 0]
    normal_data.append(record)
    episode_frames.setdefault(int(image.split("_")[0]), []).append(record)
            
    sys.stdout.write("\rLoading dataset" + "." * dots + " ")
    sys.stdout.flush()
            
    dots = dots + 1 if dots < 3 else 1

#load anomalous data set  
anomalous_data = []

camera_folder = f"{anomalous_data_folder}/camera"
imagination_folder = f"{anomalous_data_folder}/imagination"
dots = 1
        
for image in os.listdir(camera_folder):
    if not image.endswith(".npy"):
        continue
    cam_data = np.load(f"{camera_folder}/{image}", allow_pickle=True)
    img_data = np.load(f"{imagination_folder}/{image}", allow_pickle=True)
    record = [cam_data, img_data, 1]
    anomalous_data.append(record)
    episode_frames.setdefault(int(image.split("_")[0]), []).append(record)
            
    sys.stdout.write("\rLoading dataset" + "." * dots + " ")
    sys.stdout.flush()
            
    dots = dots + 1 if dots < 3 else 1

#normalized_data = normal_data[:len(anomalous_data)]        
#data_array = [*normalized_data, *anomalous_data]
data_array = [*normal_data, *anomalous_data]

#split train/validation/test by EPISODE, not by frame.
#1) sort the episode ids found in the file names, 2) cut the id list into three contiguous ranges
#   (by episode count or by frame count, see SPLIT_BY), 3) every frame (normal and anomalous) of an
#   episode goes to that episode's split, 4) shuffle the frames inside each split only.
episode_ids = sorted(episode_frames)
n_episodes = len(episode_ids)
n_frames = sum(len(episode_frames[ep]) for ep in episode_ids)

if SPLIT_BY == "episodes":
    #cut by number of episodes
    cut_1 = int(n_episodes * SPLIT_FRACTIONS[0])
    cut_2 = int(n_episodes * (SPLIT_FRACTIONS[0] + SPLIT_FRACTIONS[1]))
else:
    #cut by number of frames: keep adding whole episodes until the split holds its share of all frames
    cut_1 = None
    cut_2 = None
    cumulative_frames = 0
    for i, ep in enumerate(episode_ids):
        cumulative_frames += len(episode_frames[ep])
        if cut_1 is None and cumulative_frames >= n_frames * SPLIT_FRACTIONS[0]:
            cut_1 = i + 1
        if cut_2 is None and cumulative_frames >= n_frames * (SPLIT_FRACTIONS[0] + SPLIT_FRACTIONS[1]):
            cut_2 = i + 1
            break

train_episodes = episode_ids[:cut_1]
validation_episodes = episode_ids[cut_1:cut_2]
test_episodes = episode_ids[cut_2:]

assert train_episodes and validation_episodes and test_episodes, "a split is empty, dataset too small for SPLIT_FRACTIONS"
assert not set(train_episodes) & set(validation_episodes)
assert not set(train_episodes) & set(test_episodes)
assert not set(validation_episodes) & set(test_episodes)

def frames_of(episodes):
    frames = [frame for ep in episodes for frame in episode_frames[ep]]
    random.shuffle(frames)   #randomize the timestep frames within the split
    return frames

train_data = frames_of(train_episodes)
train_images = [image[:-1] for image in train_data]
train_labels = [label[-1] for label in train_data]

validation_data = frames_of(validation_episodes)
validation_images = [image[:-1] for image in validation_data]
validation_labels = [label[-1] for label in validation_data]
        
test_data = frames_of(test_episodes)
test_images = [image[:-1] for image in test_data]
test_labels = [label[-1] for label in test_data]

#report the split, episode ids come from the file names so gaps are possible
print()
missing_ids = sorted(set(range(episode_ids[0], episode_ids[-1] + 1)) - set(episode_ids))
print(f"Episodes found: {n_episodes} (ids {episode_ids[0]}-{episode_ids[-1]}"
      + (f", missing {missing_ids}" if missing_ids else "") + f")  frames: {n_frames}  split by: {SPLIT_BY}")
for name, episodes, frames in [("train", train_episodes, train_data), ("validation", validation_episodes, validation_data), ("test", test_episodes, test_data)]:
    anomalies = sum(frame[-1] for frame in frames)
    print(f"  {name:<10} episodes {episodes[0]:3d}-{episodes[-1]:3d} ({len(episodes):3d})  "
          f"frames {len(frames):6d} ({100 * len(frames) / n_frames:4.1f}%)  anomalous {anomalies:5d} ({100 * anomalies / max(len(frames), 1):4.1f}%)")

image_shape = train_data[0][0].shape

#log for data import      
if train_images is not None and train_labels is not None and test_images is not None and test_labels is not None:
    print("Dataset loaded successfully!!!")
    print("==============================")
    print("Train dataset length:", len(train_data))
    print("Test dateset length:", len(test_data))
    
    print("Image shape:", image_shape)
else:
    print("Dataset load failed :(")   

#quantization + restriction operators of the paper (Sec. IV-C): a (1, 1, 3, 64, 64) frame in [-0.5, 0.5] is
#turned into the 24x24 class index crop of the drivable corridor by nearest colour mapping to the CityScapes palette
def process_image(x):
    #remove batch/time dims
    x = x[0,0]
    #convert from CHW -> HWC
    x = np.transpose(x, (1,2,0))
    
    #take normalized image and turn it to color values   
    x = (x + 0.5) * 255.0
    x = np.clip(x, 0, 255).astype(np.uint8)
    
    #crop to focus erea
    x = x[34:58, 20:44]
    
    #conver bgr -> rgb
    x = x[:, :, ::-1]
    
    #map to cityscapes pallete
    h, w, _ = x.shape
    class_map = np.zeros((h,w), dtype=np.int32)
    
    colors  = np.array(list(CITYSCAPES_PALLETE.keys()))
    classes = list(CITYSCAPES_PALLETE.values())
    
    #distance based mapping, colors might not exactly match
    for i in range(h):
        for j in range(w):
            pixel = x[i,j]
            dist = np.sum((colors - pixel)**2, axis=1)
            idx = np.argmin(dist)
            
            class_map[i,j] = classes[idx]
 
    return class_map
    
#normalized class frequency histogram of a class index map
def class_histogram(seg, num_classes=20):
    hist = np.zeros(num_classes)
    valid = seg >= 0
    classes, counts = np.unique(seg[valid], return_counts=True)
    hist[classes] = counts
    hist /= hist.sum()
    
    return hist
    
#kl divergence between two class histograms
def kl_divergence_custom(p,q):
    eps = 1e-8
    return np.sum(p*np.log((p+eps)/(q+eps)))
    
#fraction of pixels whose class differs between the camera and the imagination crop
def pixel_disagreement(c, i):
    total = (c >=0) & (i >= 0)
    diff = c != i
    
    return diff[total].sum() / total.sum()
    
#class of the largest camera / imagination disagreement and the fraction of the crop it covers
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
        if perc > 0.05 and perc > obstacle[1]:
            obstacle = [obs, perc]

    return obstacle

#pre-process the data: one (camera, imagination) pair -> the (24, 24, 2) classifier input
def pre_process(data, dots = 1):
    cam = process_image(data[0])
    img = process_image(data[1])

    concat_img = np.stack((cam, img), axis=-1)
            
    return concat_img

#pre_process the data
print("Processing Images...")
x_train = np.array(list(map(pre_process, train_images)))
y_train = np.array(train_labels).reshape(-1,1)
x_validation = np.array(list(map(pre_process, validation_images)))
y_validation = np.array(validation_labels).reshape(-1,1)
x_test = np.array(list(map(pre_process, test_images)))
y_test = np.array(test_labels).reshape(-1,1)

print("Train data shape:", x_train.shape)
print("Train labels shape:", y_train.shape)

#earlier hand crafted anomaly score (kl, pixel disagreement, obstacle share) of every frame pair, kept for reference
def compute_anomaly_score(dataset):
    pixel_scores = []
    kl_scores = []
    anomaly_scores = []

    for item in dataset:
        c = item[:,:,0]
        i = item[:,:,1]
        
        px = pixel_disagreement(c, i)
        
        ch = class_histogram(c)
        ih = class_histogram(i)
        
        kl = kl_divergence_custom(ch, ih)
        
        obs = detect_obstacle(c, i)
        
        anomaly_score = kl*0.3 + px*0.4 + obs[1]*0.3
        
        pixel_scores.append(px)
        kl_scores.append(kl)
        anomaly_scores.append(anomaly_score)
    
    return anomaly_scores, pixel_scores, kl_scores
    
#accuracy maximizing threshold of a scalar score, used with compute_anomaly_score, kept for reference
def grid_search(values, labels):
    best_thresh = None
    best_acc = -1
    
    thresholds = np.unique(values)
    preds = []
    
    for thresh in thresholds:
        acc = 0
        
        for i in range(len(values)):
            v = values[i]
            l = labels[i]
            
            if v > thresh:
                preds.append(1)
                if l == 1:
                    acc += 1
            else:
                preds.append(0)
                if l == 0:
                    acc += 1
                    
        acc = acc / len(values)
        
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
            
    return best_thresh, best_acc

print("===================")
print("Computing Metrics...")

dataset = [*x_train, *x_validation, *x_test]
labels = [*y_train, *y_validation, *y_test]
#anomaly_scores, pixel_scores, kl_scores = compute_anomaly_score(dataset)

#print("===================")
#print("Computing Thresholds...")

#pixel_thresh, pixel_acc = grid_search(pixel_scores, labels)
#kl_thresh, kl_acc = grid_search(kl_scores, labels)
#anomaly_thresh, anomaly_acc = grid_search(anomaly_scores, labels)

#print("===================")
#print(f"Best Pixel threshold: {pixel_thresh:.4f}")
#print(f"Pixel Acc score: {pixel_acc:.4f}")

#print(f"Best KL threshold: {kl_thresh:.4f}")
#print(f"KL Acc score: {kl_acc:.4f}")

#print(f"Best Anomaly threshold: {anomaly_thresh:.4f}")
#print(f"Anomaly Acc score: {anomaly_acc:.4f}")
    
#roc / pr analysis of the sigmoid scores of one split, returns the selected threshold
def threshold_analysis(model, images, labels, split_name):
    from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores = model.scores(images)
    labels = np.asarray(labels).ravel().astype(int)

    fpr, tpr, roc_thresholds = roc_curve(labels, scores)
    precision, recall, pr_thresholds = precision_recall_curve(labels, scores)
    roc_auc = roc_auc_score(labels, scores)
    pr_auc = average_precision_score(labels, scores)

    #metrics of one threshold
    def at(t):
        pred = (scores >= t).astype(int)
        tp = int(((pred == 1) & (labels == 1)).sum()); fp = int(((pred == 1) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum()); tn = int(((pred == 0) & (labels == 0)).sum())
        rec = tp / max(tp + fn, 1); prec = tp / max(tp + fp, 1); fpr_t = fp / max(fp + tn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        return {"threshold": float(t), "precision": prec, "recall": rec, "fpr": fpr_t, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}

    #select the threshold on this split
    grid = np.unique(np.concatenate([roc_thresholds[np.isfinite(roc_thresholds)], np.array(CANDIDATE_THRESHOLDS)]))
    rows = [at(t) for t in grid]
    if THRESHOLD_CRITERION == "recall_at_least":
        ok = [r for r in rows if r["recall"] >= MIN_RECALL]
        selected = max(ok, key=lambda r: r["threshold"]) if ok else min(rows, key=lambda r: r["threshold"])
    elif THRESHOLD_CRITERION == "youden":
        selected = max(rows, key=lambda r: r["recall"] - r["fpr"])
    else:
        selected = max(rows, key=lambda r: r["f1"])
    youden = max(rows, key=lambda r: r["recall"] - r["fpr"])
    best_f1 = max(rows, key=lambda r: r["f1"])

    #table
    print(f"\n~~~~~~~~ Threshold analysis on the {split_name} split ~~~~~~~~")
    print(f"frames: {len(labels)}  anomalous: {int(labels.sum())} ({100 * labels.mean():.1f}%)   ROC AUC: {roc_auc:.4f}   PR AUC: {pr_auc:.4f}")
    print(f"scores in (0.01, 0.99): {100 * np.mean((scores > 0.01) & (scores < 0.99)):.1f}% of frames   (0% = perfectly bipolar output)")
    print(f"{'threshold':>10} {'precision':>10} {'recall':>8} {'fpr':>8} {'f1':>8} {'tp':>6} {'fp':>6} {'fn':>6} {'tn':>6}")
    listed = {r["threshold"]: r for r in [at(t) for t in CANDIDATE_THRESHOLDS]}
    listed[selected["threshold"]] = selected; listed[youden["threshold"]] = youden; listed[best_f1["threshold"]] = best_f1
    for r in sorted(listed.values(), key=lambda r: r["threshold"]):
        tags = []
        if r["threshold"] == selected["threshold"]: tags.append("selected")
        if r["threshold"] == youden["threshold"]: tags.append("youden")
        if r["threshold"] == best_f1["threshold"]: tags.append("max f1")
        print(f"{r['threshold']:>10.2e} {r['precision']:>10.4f} {r['recall']:>8.4f} {r['fpr']:>8.4f} {r['f1']:>8.4f} {r['tp']:>6d} {r['fp']:>6d} {r['fn']:>6d} {r['tn']:>6d}" + ("  <- " + ", ".join(tags) if tags else ""))
    print(f"criterion: {THRESHOLD_CRITERION}" + (f" (recall >= {MIN_RECALL})" if THRESHOLD_CRITERION == "recall_at_least" else ""))
    print(f"-> set CLASSIFIER_THRESHOLD = {selected['threshold']:.6g} in utils.py")

    #save the full table and the curves
    os.makedirs(os.path.dirname(THRESHOLD_TABLE_PATH), exist_ok=True)
    with open(THRESHOLD_TABLE_PATH, "w") as f:
        f.write("threshold,precision,recall,fpr,f1,tp,fp,fn,tn\n")
        for r in rows:
            f.write(f"{r['threshold']:.6g},{r['precision']:.6f},{r['recall']:.6f},{r['fpr']:.6f},{r['f1']:.6f},{r['tp']},{r['fp']},{r['fn']},{r['tn']}\n")

    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    ax[0].hist(scores[labels == 0], bins=50, range=(0, 1), alpha=0.7, label="normal", log=True)
    ax[0].hist(scores[labels == 1], bins=50, range=(0, 1), alpha=0.7, label="anomalous", log=True)
    ax[0].axvline(selected["threshold"], color="k", linestyle="--", label=f"tau = {selected['threshold']:.3g}")
    ax[0].set_xlabel("sigmoid output"); ax[0].set_ylabel("frames"); ax[0].set_title(f"score distribution ({split_name})"); ax[0].legend()
    ax[1].plot(fpr, tpr); ax[1].plot([0, 1], [0, 1], "k:", linewidth=0.8)
    ax[1].plot(selected["fpr"], selected["recall"], "ko", label=f"tau = {selected['threshold']:.3g}")
    ax[1].set_xlabel("false positive rate"); ax[1].set_ylabel("recall"); ax[1].set_title(f"ROC (AUC {roc_auc:.3f})"); ax[1].legend()
    ax[2].plot(recall, precision)
    ax[2].plot(selected["recall"], selected["precision"], "ko", label=f"tau = {selected['threshold']:.3g}")
    ax[2].set_xlabel("recall"); ax[2].set_ylabel("precision"); ax[2].set_title(f"precision-recall (AP {pr_auc:.3f})"); ax[2].legend()
    plt.tight_layout(); plt.savefig(THRESHOLD_PLOT_PATH, dpi=150); plt.close(fig)
    print(f"curves saved to {THRESHOLD_PLOT_PATH}, table to {THRESHOLD_TABLE_PATH}")
    return selected["threshold"]

model = AnomalyAutoencoder(CLASSIFIER_PATH)

if model is not None:
    print("Model imported successfully!!!")
else:
    print("Failed to import model")
    
if TRAIN_CLASSIFIER:
    model.train(x_train, y_train, x_validation, y_validation)
    model.save(CLASSIFIER_PATH)

model.evaluate(x_test, y_test)

if THRESHOLD_ANALYSIS:
    #select on the validation split, report once on the test split
    tau = threshold_analysis(model, x_validation, y_validation, "validation")
    model.threshold = tau
    test_scores = model.scores(x_test)
    test_pred = (test_scores >= tau).astype(int)
    test_truth = np.asarray(y_test).ravel().astype(int)
    tp = int(((test_pred == 1) & (test_truth == 1)).sum()); fp = int(((test_pred == 1) & (test_truth == 0)).sum())
    fn = int(((test_pred == 0) & (test_truth == 1)).sum()); tn = int(((test_pred == 0) & (test_truth == 0)).sum())
    print(f"\n~~~~~~~~ Test split at the selected threshold tau = {tau:.6g} ~~~~~~~~")
    print(f"accuracy: {(tp + tn) / max(len(test_truth), 1):.4f}  precision: {tp / max(tp + fp, 1):.4f}  recall: {tp / max(tp + fn, 1):.4f}  fpr: {fp / max(fp + tn, 1):.4f}  (tp {tp}, fp {fp}, fn {fn}, tn {tn})")

#data_idx_1 = random.randint(1, len(x_train))
#data_idx_2 = random.randint(1, len(x_validation))

#print(x_train[data_idx_1].shape)

#print("~~~~~Predict Test~~~~~")
#model.predict(np.expand_dims(x_train[data_idx_1], axis=0))
#print("Truth:", y_train[data_idx_1])

#model.predict(np.expand_dims(x_validation[data_idx_2], axis=0))
#print("Truth:", y_validation[data_idx_2])


#~~~~~~~~ non-learned baselines for system 1 (paper Sec. V-A, "Non-learned baseline") ~~~~~~~~
#the baselines see exactly the cnn's input: the same frames, the same episode-level split, the same quantization and
#24x24 corridor crop (process_image), and their threshold is picked by the same rule on the validation split, then
#frozen and applied once to the test split. all methods are scored against the same labels, so the only difference
#between them is whether the decision is learned or a plain count of disagreeing pixels.
#  A: raw semantic mismatch    -> share of crop pixels where the real class differs from the imagined class
#  B: object semantic mismatch -> as A, but only pixels whose real class is not road or sky, i.e. the disagreements
#                                 that can belong to an object on the road rather than to a blurred road / sky border
#the cnn is scored on the same split with the same rule, and once more at the paper's threshold as a consistency check
ROAD_CLASS, SKY_CLASS = 1, 11   #class indices of road and sky in CITYSCAPES_PALLETE

if MISMATCH_BASELINE:
    from sklearn.metrics import roc_auc_score, average_precision_score
    import json

    #baseline A score of every frame pair, x is (N, 24, 24, 2): [..., 0] real classes, [..., 1] imagined classes
    def mismatch_raw(x):
        return (x[..., 0] != x[..., 1]).mean(axis=(1, 2))

    #baseline B score: disagreements on object like pixels only (real class neither road nor sky)
    def mismatch_object(x):
        real, imag = x[..., 0], x[..., 1]
        disagree = (real != imag) & (real != ROAD_CLASS) & (real != SKY_CLASS)
        return disagree.mean(axis=(1, 2))

    #full metric set of a score thresholded at t (frame flagged when score >= t), plus the threshold free roc / pr areas
    def metrics_at(labels, scores, t):
        y = np.asarray(labels).ravel().astype(int)
        s = np.asarray(scores, dtype=float)
        pred = (s >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
        rec = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1); prec = tp / max(tp + fp, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = (tp * tn - fp * fn) / den if den > 0 else 0.0
        return {"threshold": float(t), "accuracy": (tp + tn) / max(len(y), 1), "balanced_accuracy": (rec + spec) / 2,
                "precision": prec, "recall": rec, "specificity": spec, "fpr": 1 - spec, "f1": f1, "mcc": mcc,
                "roc_auc": roc_auc_score(y, s), "average_precision": average_precision_score(y, s),
                "tp": tp, "fp": fp, "fn": fn, "tn": tn}

    #threshold selection on the validation split, same rule as the cnn: the largest threshold whose recall is still
    #>= MIN_RECALL. every distinct validation score is a candidate, so no grid resolution is involved
    def select_threshold(labels, scores):
        rows = [metrics_at(labels, scores, t) for t in np.unique(scores)]
        ok = [r for r in rows if r["recall"] >= MIN_RECALL]
        return max(ok, key=lambda r: r["threshold"]) if ok else min(rows, key=lambda r: r["threshold"])

    #validation and test columns side by side
    def report(name, val, test):
        keys = ["threshold", "accuracy", "balanced_accuracy", "precision", "recall", "specificity", "fpr",
                "f1", "mcc", "roc_auc", "average_precision", "tp", "fp", "fn", "tn"]
        print(f"\n~~~~~~~~ {name} ~~~~~~~~")
        print(f"{'':>18s} {'validation':>12s} {'test':>12s}")
        for k in keys:
            fmt = (lambda v: f"{v:12d}") if isinstance(val[k], int) else (lambda v: f"{v:12.4f}")
            print(f"{k:>18s} {fmt(val[k])} {fmt(test[k])}")

    results = {}

    #baselines A and B: threshold from the validation split, applied unchanged to the test split
    for name, score_fn in [("A: raw semantic mismatch", mismatch_raw), ("B: object semantic mismatch", mismatch_object)]:
        val_scores, test_scores_b = score_fn(x_validation), score_fn(x_test)
        val_m = select_threshold(y_validation, val_scores)
        test_m = metrics_at(y_test, test_scores_b, val_m["threshold"])
        report(f"{name} (validation recall >= {MIN_RECALL})", val_m, test_m)
        results[name] = (val_m, test_m)

    #cnn sigmoid scores on the same split, selected with the same rule
    cnn_val_scores, cnn_test_scores = model.scores(x_validation), model.scores(x_test)
    cnn_val = select_threshold(y_validation, cnn_val_scores)
    cnn_test = metrics_at(y_test, cnn_test_scores, cnn_val["threshold"])
    report(f"CNN (validation recall >= {MIN_RECALL})", cnn_val, cnn_test)
    results["CNN"] = (cnn_val, cnn_test)

    #cnn at the decision threshold reported in the paper (tau = 0.148534), must reproduce the paper's test figures
    paper_val = metrics_at(y_validation, cnn_val_scores, 0.148534)
    paper_test = metrics_at(y_test, cnn_test_scores, 0.148534)
    report("CNN at the paper threshold tau = 0.148534", paper_val, paper_test)
    results["CNN_paper_tau"] = (paper_val, paper_test)

    with open(MISMATCH_RESULTS_PATH, "w") as f:
        json.dump({k: {"validation": v, "test": t} for k, (v, t) in results.items()}, f, indent=2)
    print(f"\nmismatch baseline results saved to {MISMATCH_RESULTS_PATH}")
