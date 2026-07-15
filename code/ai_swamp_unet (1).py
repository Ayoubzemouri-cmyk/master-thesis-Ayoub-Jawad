# =============================================================================
# U-Net training and evaluation pipeline for landing-zone segmentation
# Extracted from ai-swamp.ipynb
# =============================================================================


# ===== Cell 1 =====
import subprocess

# TU Graz — verify image/mask folders
print("=== TU Graz semantic_drone_dataset/ ===")
out = subprocess.run(
    ["find", "/kaggle/input/datasets/bulentsiyah/semantic-drone-dataset/dataset/semantic_drone_dataset",
     "-maxdepth", "2"],
    capture_output=True, text=True
)
print(out.stdout)

# FloodNet — go deeper into Train/
print("=== FloodNet Train/ ===")
out = subprocess.run(
    ["find", "/kaggle/input/datasets/aletbm/aerial-imagery-dataset-floodnet-challenge/FloodNet Challenge - Track 1/Train",
     "-maxdepth", "3", "-type", "d"],
    capture_output=True, text=True
)
print(out.stdout)

# Sample one image + mask filename from FloodNet
print("=== FloodNet sample files ===")
out = subprocess.run(
    ["find", "/kaggle/input/datasets/aletbm/aerial-imagery-dataset-floodnet-challenge/FloodNet Challenge - Track 1/Train",
     "-type", "f", "(", "-name", "*.jpg", "-o", "-name", "*.png", ")"],
    capture_output=True, text=True
)
files = out.stdout.strip().split("\n")
print(f"Total files: {len(files)}")
for f in files[:6]:
    print(f"  {f}")


# ===== Cell 2 =====
# =============================================================================
# CELL 1 — Setup, data loading, and train/val/test split  (FIXED PATHS)
# =============================================================================

import os
import json
import numpy as np
from PIL import Image
from glob import glob
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -----------------------------------------------------------------------------
# Paths  (CORRECTED — datasets are under /kaggle/input/datasets/...)
# -----------------------------------------------------------------------------
WORK_DIR = "/kaggle/working"
os.makedirs(WORK_DIR, exist_ok=True)

X_CACHE = os.path.join(WORK_DIR, "X_data.npy")
Y_CACHE = os.path.join(WORK_DIR, "Y_data.npy")

TUGRAZ_IMG_DIR = "/kaggle/input/datasets/bulentsiyah/semantic-drone-dataset/dataset/semantic_drone_dataset/original_images"
TUGRAZ_MSK_DIR = "/kaggle/input/datasets/bulentsiyah/semantic-drone-dataset/dataset/semantic_drone_dataset/label_images_semantic"

FLOODNET_BASE = "/kaggle/input/datasets/aletbm/aerial-imagery-dataset-floodnet-challenge/FloodNet Challenge - Track 1/Train/Labeled"

IMG_SIZE = 256

TUGRAZ_SAFE_CLASSES   = {1, 2, 3, 4, 5}        # paved, dirt, grass, gravel, water
FLOODNET_SAFE_CLASSES = {4, 5, 9}              # road-non-flooded, water, grass


# -----------------------------------------------------------------------------
# Loaders
# -----------------------------------------------------------------------------
def load_image(path, size=IMG_SIZE):
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0

def load_mask_binary(path, safe_classes, size=IMG_SIZE):
    msk = Image.open(path).resize((size, size), Image.NEAREST)
    msk = np.asarray(msk, dtype=np.uint8)
    if msk.ndim == 3:
        msk = msk[..., 0]
    binary = np.isin(msk, list(safe_classes)).astype(np.float32)
    return binary[..., np.newaxis]


def load_tugraz():
    img_paths = sorted(glob(os.path.join(TUGRAZ_IMG_DIR, "*.jpg")))
    X, Y, missing = [], [], 0
    for ip in img_paths:
        stem = os.path.splitext(os.path.basename(ip))[0]
        mp = os.path.join(TUGRAZ_MSK_DIR, stem + ".png")
        if not os.path.exists(mp):
            missing += 1
            continue
        X.append(load_image(ip))
        Y.append(load_mask_binary(mp, TUGRAZ_SAFE_CLASSES))
    print(f"  TU Graz: {len(X)} pairs loaded ({missing} missing masks)")
    return X, Y


def load_floodnet():
    X, Y, missing = [], [], 0
    for split in ["Flooded", "Non-Flooded"]:
        img_dir  = os.path.join(FLOODNET_BASE, split, "image")
        mask_dir = os.path.join(FLOODNET_BASE, split, "mask")
        img_paths = sorted(glob(os.path.join(img_dir, "*.jpg")))
        for ip in img_paths:
            stem = os.path.splitext(os.path.basename(ip))[0]
            mp = os.path.join(mask_dir, stem + "_lab.png")
            if not os.path.exists(mp):
                missing += 1
                continue
            X.append(load_image(ip))
            Y.append(load_mask_binary(mp, FLOODNET_SAFE_CLASSES))
    print(f"  FloodNet: {len(X)} pairs loaded ({missing} missing masks)")
    return X, Y


# -----------------------------------------------------------------------------
# Cache invalidation: if existing .npy files are empty, ignore them
# -----------------------------------------------------------------------------
def cache_is_valid():
    if not (os.path.exists(X_CACHE) and os.path.exists(Y_CACHE)):
        return False
    try:
        x = np.load(X_CACHE, mmap_mode="r")
        return x.shape[0] > 0
    except Exception:
        return False

if cache_is_valid():
    print("Loading from cache...")
    X_data = np.load(X_CACHE)
    Y_data = np.load(Y_CACHE)
else:
    # Remove empty/broken cache if present
    for p in (X_CACHE, Y_CACHE):
        if os.path.exists(p):
            os.remove(p)

    print("Loading TU Graz...")
    X1, Y1 = load_tugraz()
    print("Loading FloodNet...")
    X2, Y2 = load_floodnet()

    X_data = np.array(X1 + X2, dtype=np.float32)
    Y_data = np.array(Y1 + Y2, dtype=np.float32)

    print(f"Caching to {X_CACHE} and {Y_CACHE}...")
    np.save(X_CACHE, X_data)
    np.save(Y_CACHE, Y_data)

print(f"\nTotal: {len(X_data)} image-mask pairs")
print(f"X shape: {X_data.shape}, dtype: {X_data.dtype}")
print(f"Y shape: {Y_data.shape}, dtype: {Y_data.dtype}")
print(f"Class balance: {Y_data.mean()*100:.1f}% safe, {(1-Y_data.mean())*100:.1f}% unsafe")


# -----------------------------------------------------------------------------
# Split 80 / 10 / 10
# -----------------------------------------------------------------------------
X_trainval, X_test, Y_trainval, Y_test = train_test_split(
    X_data, Y_data, test_size=0.10, random_state=SEED)
X_train, X_val, Y_train, Y_val = train_test_split(
    X_trainval, Y_trainval, test_size=0.1111, random_state=SEED)

print(f"\nTrain: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")


# ===== Cell 3 =====
# =============================================================================
# CELL 2 — U-Net architecture and IoU metric
# =============================================================================
# Defines the same ~7.77M-parameter U-Net from your previous runs and the
# custom binary IoU metric. The metric is registered as a Keras serializable
# so it survives save/load round-trips when resuming.
# =============================================================================

@tf.keras.utils.register_keras_serializable()
class BinaryIoU(tf.keras.metrics.Metric):
    """IoU for binary segmentation. Threshold at 0.5."""
    def __init__(self, name="iou", **kwargs):
        super().__init__(name=name, **kwargs)
        self.intersection = self.add_weight(name="i", initializer="zeros")
        self.union        = self.add_weight(name="u", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred > 0.5, tf.float32)
        inter = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - inter
        self.intersection.assign_add(inter)
        self.union.assign_add(union)

    def result(self):
        return self.intersection / (self.union + 1e-7)

    def reset_state(self):
        self.intersection.assign(0.0)
        self.union.assign(0.0)


def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def build_unet(input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    inp = layers.Input(input_shape)

    # Encoder
    c1 = conv_block(inp, 32);  p1 = layers.MaxPool2D()(c1)
    c2 = conv_block(p1,  64);  p2 = layers.MaxPool2D()(c2)
    c3 = conv_block(p2, 128);  p3 = layers.MaxPool2D()(c3)
    c4 = conv_block(p3, 256);  p4 = layers.MaxPool2D()(c4)

    # Bottleneck
    b  = conv_block(p4, 512)

    # Decoder with skip connections
    u4 = layers.Conv2DTranspose(256, 2, strides=2, padding="same")(b)
    u4 = layers.Concatenate()([u4, c4]);  u4 = conv_block(u4, 256)
    u3 = layers.Conv2DTranspose(128, 2, strides=2, padding="same")(u4)
    u3 = layers.Concatenate()([u3, c3]);  u3 = conv_block(u3, 128)
    u2 = layers.Conv2DTranspose( 64, 2, strides=2, padding="same")(u3)
    u2 = layers.Concatenate()([u2, c2]);  u2 = conv_block(u2,  64)
    u1 = layers.Conv2DTranspose( 32, 2, strides=2, padding="same")(u2)
    u1 = layers.Concatenate()([u1, c1]);  u1 = conv_block(u1,  32)

    out = layers.Conv2D(1, 1, activation="sigmoid")(u1)
    return models.Model(inp, out, name="unet_binary_safety")


# Quick sanity check (don't compile yet — Cell 3 handles compile/load)
_tmp = build_unet()
print(f"U-Net params: {_tmp.count_params():,}")
del _tmp


# ===== Cell 4 =====
# =============================================================================
# CELL 3 — Training with full crash resume
# =============================================================================
# This cell trains to TARGET_EPOCHS total. If interrupted and re-run, it
# resumes from the last completed epoch — no progress lost.
#
# Files written to /kaggle/working/ after every epoch:
#   - last_checkpoint.keras   (full model, overwritten each epoch)
#   - best_model.keras        (best by val_iou, only updated when improved)
#   - history.json            (cumulative per-epoch metrics)
#   - epoch_state.json        (last_completed_epoch)
# =============================================================================

import math

TARGET_EPOCHS = 100
BATCH_SIZE    = 8
INIT_LR       = 1e-4

LAST_CKPT  = os.path.join(WORK_DIR, "last_checkpoint.keras")
BEST_CKPT  = os.path.join(WORK_DIR, "best_model.keras")
HIST_PATH  = os.path.join(WORK_DIR, "history.json")
STATE_PATH = os.path.join(WORK_DIR, "epoch_state.json")


# -----------------------------------------------------------------------------
# Augmentation as a tf.data pipeline (flips + brightness ±10%)
# -----------------------------------------------------------------------------
def augment(img, msk):
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_left_right(img); msk = tf.image.flip_left_right(msk)
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_up_down(img);    msk = tf.image.flip_up_down(msk)
    img = tf.image.random_brightness(img, 0.1)
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img, msk

train_ds = (tf.data.Dataset.from_tensor_slices((X_train, Y_train))
            .shuffle(512, seed=SEED)
            .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))
val_ds   = (tf.data.Dataset.from_tensor_slices((X_val, Y_val))
            .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))


# -----------------------------------------------------------------------------
# Resume logic
# -----------------------------------------------------------------------------
custom_objs = {"BinaryIoU": BinaryIoU, "iou": BinaryIoU}

if os.path.exists(LAST_CKPT) and os.path.exists(STATE_PATH):
    print("Resuming from checkpoint...")
    unet = tf.keras.models.load_model(LAST_CKPT, custom_objects=custom_objs)
    with open(STATE_PATH) as f:
        initial_epoch = json.load(f)["last_completed_epoch"]
    with open(HIST_PATH) as f:
        history_log = json.load(f)
    print(f"  Resuming at epoch {initial_epoch} / {TARGET_EPOCHS}")
else:
    print("Starting fresh training run...")
    unet = build_unet()
    unet.compile(optimizer=optimizers.Adam(INIT_LR),
                 loss="binary_crossentropy",
                 metrics=["accuracy", BinaryIoU(name="iou")])
    initial_epoch = 0
    history_log = {"loss": [], "accuracy": [], "iou": [],
                   "val_loss": [], "val_accuracy": [], "val_iou": [], "lr": []}


# -----------------------------------------------------------------------------
# Per-epoch save callback (the heart of the resume system)
# -----------------------------------------------------------------------------
class EpochSaver(callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.best_val_iou = max(history_log["val_iou"]) if history_log["val_iou"] else -1.0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Append to history
        for k in ["loss", "accuracy", "iou", "val_loss", "val_accuracy", "val_iou"]:
            history_log[k].append(float(logs.get(k, float("nan"))))
        history_log["lr"].append(float(self.model.optimizer.learning_rate.numpy()))

        # Save: full checkpoint, history, epoch state
        self.model.save(LAST_CKPT)
        with open(HIST_PATH,  "w") as f: json.dump(history_log, f)
        with open(STATE_PATH, "w") as f: json.dump({"last_completed_epoch": epoch + 1}, f)

        # Best-by-val-iou
        v = float(logs.get("val_iou", -1))
        if v > self.best_val_iou:
            self.best_val_iou = v
            self.model.save(BEST_CKPT)
            print(f"  ↳ saved best (val_iou={v:.4f})")


cbs = [
    EpochSaver(),
    callbacks.ReduceLROnPlateau(monitor="val_iou", mode="max",
                                factor=0.5, patience=5, min_lr=1e-7, verbose=1),
]


# -----------------------------------------------------------------------------
# Train
# -----------------------------------------------------------------------------
if initial_epoch < TARGET_EPOCHS:
    unet.fit(train_ds,
             validation_data=val_ds,
             epochs=TARGET_EPOCHS,
             initial_epoch=initial_epoch,
             callbacks=cbs,
             verbose=1)
else:
    print(f"Already trained to {initial_epoch} epochs — skipping.")


# ===== Cell 5 =====
# =============================================================================
# CELL 4 — Evaluation and thesis figure generation
# =============================================================================
# Loads the best model, evaluates on test set, and saves 7 individual PNGs:
#   thesis_loss.png, thesis_accuracy.png, thesis_iou.png,
#   thesis_example_1..4.png  (each: input | ground truth | prediction)
# =============================================================================

import matplotlib.pyplot as plt

# Load best model + history
unet_best = tf.keras.models.load_model(BEST_CKPT, custom_objects=custom_objs)
with open(HIST_PATH) as f:
    hist = json.load(f)

# -----------------------------------------------------------------------------
# Test-set metrics
# -----------------------------------------------------------------------------
test_ds = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(BATCH_SIZE)
test_loss, test_acc, test_iou = unet_best.evaluate(test_ds, verbose=0)
print(f"Test  loss: {test_loss:.4f}")
print(f"Test  acc:  {test_acc*100:.2f}%")
print(f"Test  IoU:  {test_iou:.4f}")

# Per-class IoU on test set
preds = (unet_best.predict(X_test, batch_size=BATCH_SIZE, verbose=0) > 0.5).astype(np.float32)
def iou_class(y_true, y_pred, cls):
    yt = (y_true == cls).astype(np.float32)
    yp = (y_pred == cls).astype(np.float32)
    inter = (yt * yp).sum()
    union = yt.sum() + yp.sum() - inter
    return inter / (union + 1e-7)
print(f"Safe   IoU: {iou_class(Y_test, preds, 1):.4f}")
print(f"Unsafe IoU: {iou_class(Y_test, preds, 0):.4f}")


# -----------------------------------------------------------------------------
# Training curves — one PNG each
# -----------------------------------------------------------------------------
def save_curve(metric_train, metric_val, ylabel, fname):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(hist[metric_train], label="Train")
    ax.plot(hist[metric_val],   label="Validation")
    ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(WORK_DIR, fname)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")

save_curve("loss",     "val_loss",     "Binary cross-entropy loss", "thesis_loss.png")
save_curve("accuracy", "val_accuracy", "Pixel accuracy",            "thesis_accuracy.png")
save_curve("iou",      "val_iou",      "IoU",                       "thesis_iou.png")


# -----------------------------------------------------------------------------
# Test examples — pick 4 spread across the test set
# -----------------------------------------------------------------------------
indices = np.linspace(0, len(X_test) - 1, 4, dtype=int)

for n, idx in enumerate(indices, start=1):
    img  = X_test[idx]
    gt   = Y_test[idx, ..., 0]
    pred = preds[idx, ..., 0]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    axes[0].imshow(img);                       axes[0].set_title("Input image")
    axes[1].imshow(gt,   cmap="gray", vmin=0, vmax=1); axes[1].set_title("Ground truth (safe=white)")
    axes[2].imshow(pred, cmap="gray", vmin=0, vmax=1); axes[2].set_title("U-Net prediction")
    for ax in axes: ax.axis("off")
    fig.tight_layout()

    out = os.path.join(WORK_DIR, f"thesis_example_{n}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")

print("\nAll thesis figures saved to /kaggle/working/")


# ===== Cell 6 =====
# =============================================================================
# CELL 5 — All recommended thesis figures
# =============================================================================
# Generates the full set of recommended thesis figures for the U-Net.
# Requires Cells 1 and 2 to have run (uses X_test, Y_test, BinaryIoU, build_unet,
# WORK_DIR, BEST_CKPT). All output PNGs saved to /kaggle/working/.
#
# Generates:
#   thesis_dataset_balance.png            — class balance bar chart
#   thesis_confusion_matrix.png           — pixel-level confusion matrix
#   thesis_pr_curve.png                   — precision-recall curve
#   thesis_iou_vs_threshold.png           — IoU as a function of threshold
#   thesis_per_dataset_iou.png            — IoU on TU Graz vs FloodNet test pixels
#   thesis_error_map_1.png                — input/GT/pred/error 4-panel
#   thesis_error_map_2.png                — same, second example
#   thesis_results_table.txt              — final metrics table (text)
# =============================================================================

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

# -----------------------------------------------------------------------------
# Reload best model + raw test predictions (probabilities, not thresholded)
# -----------------------------------------------------------------------------
custom_objs = {"BinaryIoU": BinaryIoU, "iou": BinaryIoU}
unet_best = tf.keras.models.load_model(BEST_CKPT, custom_objects=custom_objs)

probs = unet_best.predict(X_test, batch_size=8, verbose=0)   # (N, 256, 256, 1) in [0,1]
preds = (probs > 0.5).astype(np.float32)
y_true = Y_test
y_true_flat = y_true.flatten()
y_pred_flat = preds.flatten()
prob_flat   = probs.flatten()


# -----------------------------------------------------------------------------
# 1. Class balance bar chart  (whole combined dataset)
# -----------------------------------------------------------------------------
safe_frac   = float(Y_data.mean())
unsafe_frac = 1.0 - safe_frac

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(["Safe", "Unsafe"], [safe_frac, unsafe_frac],
              color=["#4C9F70", "#C44536"])
for bar, frac in zip(bars, [safe_frac, unsafe_frac]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{frac*100:.1f}%", ha="center", fontsize=11)
ax.set_ylabel("Fraction of pixels")
ax.set_ylim(0, 1.0)
ax.set_title("Combined dataset — pixel-level class balance (798 images)")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(WORK_DIR, "thesis_dataset_balance.png"), dpi=150)
plt.close(fig)
print("  saved thesis_dataset_balance.png")


# -----------------------------------------------------------------------------
# 2. Confusion matrix (pixel-level, on test set)
# -----------------------------------------------------------------------------
TP = int(((y_true_flat == 1) & (y_pred_flat == 1)).sum())
TN = int(((y_true_flat == 0) & (y_pred_flat == 0)).sum())
FP = int(((y_true_flat == 0) & (y_pred_flat == 1)).sum())  # predicted safe but actually unsafe — DANGEROUS
FN = int(((y_true_flat == 1) & (y_pred_flat == 0)).sum())  # predicted unsafe but actually safe — overly cautious
total = TP + TN + FP + FN

cm = np.array([[TN, FP],
               [FN, TP]])
cm_pct = cm / total * 100

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=cm_pct.max())
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["Pred: Unsafe", "Pred: Safe"])
ax.set_yticklabels(["True: Unsafe", "True: Safe"])
ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
ax.set_title("Pixel-level confusion matrix (test set)")
for i in range(2):
    for j in range(2):
        # Pick text colour automatically for legibility against the heatmap
        color = "white" if cm_pct[i, j] > cm_pct.max() / 2 else "black"
        ax.text(j, i, f"{cm[i,j]:,}\n({cm_pct[i,j]:.1f}%)",
                ha="center", va="center", color=color, fontsize=11)
plt.colorbar(im, ax=ax, label="% of test pixels")
fig.tight_layout()
fig.savefig(os.path.join(WORK_DIR, "thesis_confusion_matrix.png"), dpi=150)
plt.close(fig)
print("  saved thesis_confusion_matrix.png")
print(f"    TP={TP:,}  TN={TN:,}  FP={FP:,}  FN={FN:,}")
print(f"    FP rate (dangerous: predicted safe but actually unsafe): {FP/(FP+TN)*100:.2f}%")
print(f"    FN rate (cautious:  predicted unsafe but actually safe): {FN/(FN+TP)*100:.2f}%")


# -----------------------------------------------------------------------------
# 3. Precision-recall curve  (with respect to safe class)
# -----------------------------------------------------------------------------
# Subsample to keep the sklearn call tractable (5.2M pixels otherwise)
rng = np.random.default_rng(42)
n_sample = 500_000
idx = rng.choice(prob_flat.size, size=n_sample, replace=False)

from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score
prec, rec, pr_thresh = precision_recall_curve(y_true_flat[idx], prob_flat[idx])
ap = average_precision_score(y_true_flat[idx], prob_flat[idx])

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(rec, prec, color="tab:blue", linewidth=2)
# Mark the operating point at threshold = 0.5
i_05 = np.argmin(np.abs(pr_thresh - 0.5))
ax.scatter(rec[i_05], prec[i_05], color="tab:red", s=60, zorder=5,
           label=f"Threshold = 0.5  (P={prec[i_05]:.3f}, R={rec[i_05]:.3f})")
ax.set_xlabel("Recall (safe class)")
ax.set_ylabel("Precision (safe class)")
ax.set_title(f"Precision-recall curve  —  AP = {ap:.4f}")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
ax.grid(alpha=0.3); ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(WORK_DIR, "thesis_pr_curve.png"), dpi=150)
plt.close(fig)
print(f"  saved thesis_pr_curve.png  (AP = {ap:.4f})")


# -----------------------------------------------------------------------------
# 4. IoU as a function of threshold
# -----------------------------------------------------------------------------
thresholds = np.linspace(0.05, 0.95, 19)
iou_safe, iou_unsafe, miou = [], [], []
for t in thresholds:
    p = (probs > t).astype(np.float32).flatten()
    # Safe IoU
    inter = ((y_true_flat == 1) & (p == 1)).sum()
    union = ((y_true_flat == 1) | (p == 1)).sum()
    s = inter / (union + 1e-7)
    # Unsafe IoU
    inter2 = ((y_true_flat == 0) & (p == 0)).sum()
    union2 = ((y_true_flat == 0) | (p == 0)).sum()
    u = inter2 / (union2 + 1e-7)
    iou_safe.append(s); iou_unsafe.append(u); miou.append((s + u) / 2)

best_t_idx = int(np.argmax(miou))
best_t = thresholds[best_t_idx]

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(thresholds, iou_safe,   label="Safe IoU",   color="#4C9F70", linewidth=2)
ax.plot(thresholds, iou_unsafe, label="Unsafe IoU", color="#C44536", linewidth=2)
ax.plot(thresholds, miou,       label="Mean IoU",   color="black", linewidth=2, linestyle="--")
ax.axvline(0.5, color="gray", linestyle=":", label="Default threshold (0.5)")
ax.axvline(best_t, color="purple", linestyle=":",
           label=f"Optimal threshold ({best_t:.2f}, mIoU={miou[best_t_idx]:.4f})")
ax.set_xlabel("Decision threshold")
ax.set_ylabel("IoU")
ax.set_title("IoU vs. decision threshold")
ax.legend(loc="lower center"); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(WORK_DIR, "thesis_iou_vs_threshold.png"), dpi=150)
plt.close(fig)
print(f"  saved thesis_iou_vs_threshold.png  (best threshold = {best_t:.2f})")


# -----------------------------------------------------------------------------
# 5. Per-dataset IoU breakdown
# -----------------------------------------------------------------------------
# Re-derive which test indices come from TU Graz (first 400) vs FloodNet (last 398)
# We need to know which positions in X_data fell into the test split.
# Easiest: re-run the same split with the same seed and track origins.
n_tugraz   = 400
n_floodnet = 398
origins_full = np.array(["tugraz"] * n_tugraz + ["floodnet"] * n_floodnet)

# Reproduce the split using sklearn — same seed, same fractions
indices = np.arange(len(X_data))
idx_trainval, idx_test = train_test_split(indices, test_size=0.10, random_state=SEED)
test_origins = origins_full[idx_test]

print(f"\n  Test split breakdown: "
      f"{(test_origins == 'tugraz').sum()} TU Graz, "
      f"{(test_origins == 'floodnet').sum()} FloodNet")

def iou_for_subset(mask_indices):
    if mask_indices.sum() == 0:
        return None, None, None
    yt = Y_test[mask_indices].flatten()
    yp = preds[mask_indices].flatten()
    # Safe
    s_inter = ((yt == 1) & (yp == 1)).sum(); s_union = ((yt == 1) | (yp == 1)).sum()
    s = s_inter / (s_union + 1e-7)
    # Unsafe
    u_inter = ((yt == 0) & (yp == 0)).sum(); u_union = ((yt == 0) | (yp == 0)).sum()
    u = u_inter / (u_union + 1e-7)
    return s, u, (s + u) / 2

s_tg, u_tg, m_tg = iou_for_subset(test_origins == "tugraz")
s_fn, u_fn, m_fn = iou_for_subset(test_origins == "floodnet")
s_all, u_all, m_all = iou_for_subset(np.ones(len(test_origins), dtype=bool))

groups   = ["TU Graz", "FloodNet", "Combined"]
safe_v   = [s_tg, s_fn, s_all]
unsafe_v = [u_tg, u_fn, u_all]
miou_v   = [m_tg, m_fn, m_all]

x = np.arange(len(groups)); width = 0.27
fig, ax = plt.subplots(figsize=(7, 5))
b1 = ax.bar(x - width, safe_v,   width, label="Safe IoU",   color="#4C9F70")
b2 = ax.bar(x,         unsafe_v, width, label="Unsafe IoU", color="#C44536")
b3 = ax.bar(x + width, miou_v,   width, label="Mean IoU",   color="#5B6B8A")
for bars in (b1, b2, b3):
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(groups)
ax.set_ylabel("IoU"); ax.set_ylim(0, 1.0)
ax.set_title("Test-set IoU by source dataset")
ax.legend(); ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(WORK_DIR, "thesis_per_dataset_iou.png"), dpi=150)
plt.close(fig)
print("  saved thesis_per_dataset_iou.png")


# -----------------------------------------------------------------------------
# 6. Error maps (4-panel: input / GT / pred / error visualization)
# -----------------------------------------------------------------------------
# Pick two indices that produce informative error patterns
# (one from each dataset if possible)
err_indices = []
for origin in ["tugraz", "floodnet"]:
    cands = np.where(test_origins == origin)[0]
    if len(cands) > 0:
        # Pick the one with the most balanced unsafe presence
        unsafe_counts = [(1 - Y_test[i]).sum() for i in cands]
        # Choose median so we don't get an all-safe or all-unsafe trivial case
        chosen = cands[np.argsort(unsafe_counts)[len(cands)//2]]
        err_indices.append(chosen)

# Custom 4-color colormap: TN=light gray, TP=white, FP=red (DANGEROUS), FN=blue (cautious)
err_cmap = mcolors.ListedColormap(["#DDDDDD", "#FFFFFF", "#C44536", "#4A6FA5"])
err_norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], err_cmap.N)
# Encoding:
#   0 = TN (true unsafe, predicted unsafe)         — light gray
#   1 = TP (true safe, predicted safe)             — white
#   2 = FP (true unsafe, predicted safe — DANGER)  — red
#   3 = FN (true safe, predicted unsafe — caution) — blue

for n, idx in enumerate(err_indices, start=1):
    img  = X_test[idx]
    gt   = Y_test[idx, ..., 0]
    pred = preds[idx, ..., 0]

    err = np.zeros_like(gt, dtype=np.int32)
    err[(gt == 0) & (pred == 0)] = 0  # TN
    err[(gt == 1) & (pred == 1)] = 1  # TP
    err[(gt == 0) & (pred == 1)] = 2  # FP — dangerous
    err[(gt == 1) & (pred == 0)] = 3  # FN — cautious

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    axes[0].imshow(img);                                axes[0].set_title("Input image")
    axes[1].imshow(gt,   cmap="gray", vmin=0, vmax=1);  axes[1].set_title("Ground truth")
    axes[2].imshow(pred, cmap="gray", vmin=0, vmax=1);  axes[2].set_title("Prediction")
    axes[3].imshow(err, cmap=err_cmap, norm=err_norm);  axes[3].set_title("Error map")
    for ax in axes: ax.axis("off")

    legend_handles = [
        mpatches.Patch(color="#FFFFFF", label="True Positive (correct safe)",   ec="black"),
        mpatches.Patch(color="#DDDDDD", label="True Negative (correct unsafe)", ec="black"),
        mpatches.Patch(color="#C44536", label="False Positive (DANGER: predicted safe, actually unsafe)"),
        mpatches.Patch(color="#4A6FA5", label="False Negative (cautious: predicted unsafe, actually safe)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.05), fontsize=9)
    fig.tight_layout()
    out = os.path.join(WORK_DIR, f"thesis_error_map_{n}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}  (origin: {test_origins[idx]})")


# -----------------------------------------------------------------------------
# 7. Final results table (text file you can paste into LaTeX/Word)
# -----------------------------------------------------------------------------
table = f"""
=== U-Net binary safety segmentation — final test results ===

Dataset:          798 images (400 TU Graz + 398 FloodNet)
Resolution:       256 x 256
Train/Val/Test:   638 / 80 / 80
Class balance:    {safe_frac*100:.1f}% safe, {unsafe_frac*100:.1f}% unsafe
Architecture:     U-Net (32-64-128-256-512 filters, ~7.77M params)
Training:         100 epochs, Adam (lr=1e-4), BCE loss, ReduceLROnPlateau
Augmentation:     horizontal+vertical flips, brightness +-10%

Best validation IoU: {reg_bv if 'reg_bv' in dir() else 'see history.json'}

--- Test set ---
Pixel accuracy:           {((y_true_flat == y_pred_flat).mean())*100:.2f}%
IoU (safe class):         {s_all:.4f}
IoU (unsafe class):       {u_all:.4f}
Mean IoU:                 {m_all:.4f}
Average Precision (PR):   {ap:.4f}
Optimal threshold:        {best_t:.2f}  (mIoU at this threshold: {miou[best_t_idx]:.4f})

--- Per-dataset IoU ---
TU Graz   — safe: {s_tg:.4f}  unsafe: {u_tg:.4f}  mean: {m_tg:.4f}
FloodNet  — safe: {s_fn:.4f}  unsafe: {u_fn:.4f}  mean: {m_fn:.4f}

--- Pixel-level error breakdown ---
True Positives  (correct safe):    {TP:>10,}  ({TP/total*100:.2f}%)
True Negatives  (correct unsafe):  {TN:>10,}  ({TN/total*100:.2f}%)
False Positives (DANGEROUS):       {FP:>10,}  ({FP/total*100:.2f}%)
False Negatives (overly cautious): {FN:>10,}  ({FN/total*100:.2f}%)

False positive rate:  {FP/(FP+TN)*100:.2f}%
False negative rate:  {FN/(FN+TP)*100:.2f}%
"""
table_path = os.path.join(WORK_DIR, "thesis_results_table.txt")
with open(table_path, "w") as f:
    f.write(table)
print(table)
print(f"  saved {table_path}")
print("\nAll thesis figures + table saved to /kaggle/working/")
