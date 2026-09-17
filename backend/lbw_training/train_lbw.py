"""
Train the multi-head LBW sub-component model (Pitching / Impact / Wickets).

Run (from inside backend/lbw_training/):
    pip install torch torchvision opencv-python numpy
    python train_lbw.py

Produces: ../models/lbw_submodel.pt
The FastAPI backend (app.py) will automatically pick it up on next restart.

--------------------------------------------------------------------------
IMPORTANT — read before you present this:
- Only 19 labeled clips total (v1-v19; v20 had no readable ground-truth
  overlay so it was excluded). This is NOT enough data to train a model
  that generalizes to new real-world footage. Treat this as a proof-of-
  concept / demo showing the pipeline works end-to-end, not a validated
  production model. Say this plainly if asked in your viva/defense.
- Labels were extracted via OCR from the on-screen graphic burned into
  the second half of each clip (Pitching/Impact/Wickets/Decision boxes),
  cross-checked by eye where OCR was ambiguous.
- To stop the model from "cheating" by reading that on-screen text, only
  the FIRST 55% of each clip's frames are used for training (run-up
  through impact, before the graphic appears).
- The final OUT/NOT-OUT verdict is computed by a hand-coded rule engine
  from the (pitching, impact, wickets) predictions -- exactly matching
  real LBW law -- not by a black-box "decision" head. This is more
  explainable and more defensible than an end-to-end classifier here.
--------------------------------------------------------------------------
"""
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.video import r3d_18

HERE = Path(__file__).parent
DATA_DIR = HERE / "dataset"
LABELS_CSV = HERE / "labels.csv"
OUT_PATH = HERE.parent / "models" / "lbw_submodel.pt"

CLIP_LEN = 16
RESIZE = 112
MEAN = [0.43216, 0.394666, 0.37645]
STD = [0.22803, 0.22145, 0.216989]
PRE_GRAPHIC_FRACTION = 0.55

PITCHING_CLASSES = ["OUTSIDE_LEG", "OUTSIDE_OFF", "IN_LINE"]
IMPACT_CLASSES = ["OUTSIDE", "IN_LINE"]
WICKETS_CLASSES = ["MISSING", "HITTING"]
DECISION_CLASSES = ["NOT_OUT", "OUT"]


def sample_frame_indices(total_frames, clip_len):
    usable = max(int(total_frames * PRE_GRAPHIC_FRACTION), clip_len)
    if usable >= clip_len:
        return np.linspace(0, usable - 1, clip_len).astype(int)
    reps = int(np.ceil(clip_len / usable))
    return np.tile(np.arange(usable), reps)[:clip_len]


def load_clip(path, clip_len=CLIP_LEN, resize=RESIZE):
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = sample_frame_indices(total, clip_len)
    wanted = set(indices.tolist())
    frames = {}
    idx = 0
    while cap.isOpened() and len(frames) < len(wanted):
        ret, frame = cap.read()
        if not ret:
            break
        if idx in wanted:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (resize, resize))
            frames[idx] = frame
        idx += 1
    cap.release()
    last = np.zeros((resize, resize, 3), dtype=np.uint8)
    clip = []
    for i in indices:
        if i in frames:
            last = frames[i]
        clip.append(last)
    return np.stack(clip)


class LBWDataset(Dataset):
    def __init__(self, rows, augment=False):
        self.rows = rows
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        clip = load_clip(DATA_DIR / f"{row['video']}.mp4")
        if self.augment and np.random.rand() < 0.5:
            factor = 0.8 + np.random.rand() * 0.4
            clip = np.clip(clip.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        clip = clip.astype(np.float32) / 255.0
        clip = (clip - MEAN) / STD
        clip = np.ascontiguousarray(clip.transpose(3, 0, 1, 2))
        tensor = torch.from_numpy(clip).float()

        def idx_or_ignore(classes, val):
            return classes.index(val) if val != "UNKNOWN" else -100

        y_pitch = idx_or_ignore(PITCHING_CLASSES, row["pitching"])
        y_impact = idx_or_ignore(IMPACT_CLASSES, row["impact"])
        y_wkts = idx_or_ignore(WICKETS_CLASSES, row["wickets"])
        y_decision = DECISION_CLASSES.index(row["decision"])
        return tensor, y_pitch, y_impact, y_wkts, y_decision


class MultiHeadLBW(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = r3d_18(weights="DEFAULT")
        for p in backbone.parameters():
            p.requires_grad = False
        for p in backbone.layer4.parameters():
            p.requires_grad = True
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head_pitch = nn.Linear(in_features, len(PITCHING_CLASSES))
        self.head_impact = nn.Linear(in_features, len(IMPACT_CLASSES))
        self.head_wkts = nn.Linear(in_features, len(WICKETS_CLASSES))
        self.head_decision = nn.Linear(in_features, len(DECISION_CLASSES))  # aux only

    def forward(self, x):
        feat = self.backbone(x)
        return (
            self.head_pitch(feat),
            self.head_impact(feat),
            self.head_wkts(feat),
            self.head_decision(feat),
        )


def safe_ce(criterion, output, target):
    """Cross-entropy that safely returns 0 (not NaN) when every target in
    the batch is the ignore_index -- avoids corrupting shared backbone
    weights when a batch happens to have no valid labels for this head."""
    valid = (target != -100).sum()
    if valid.item() == 0:
        return torch.tensor(0.0, device=output.device, requires_grad=True)
    return criterion(output, target)


def main():
    rows = list(csv.DictReader(open(LABELS_CSV)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {len(rows)} clips, device={device}")

    train_ds = LBWDataset(rows, augment=True)
    # Full-batch: with only 19 clips, small random batches can sometimes end up
    # with ALL "UNKNOWN" labels for a head, which previously produced a NaN loss
    # and corrupted the shared backbone weights. Using the whole dataset as one
    # batch guarantees every head sees its valid-labeled samples every step.
    train_loader = DataLoader(train_ds, batch_size=len(rows), shuffle=True, num_workers=0)

    model = MultiHeadLBW().to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)

    EPOCHS = 30
    model.train()
    for epoch in range(EPOCHS):
        total_loss, correct_dec, n = 0.0, 0, 0
        for clips, y_p, y_i, y_w, y_d in train_loader:
            clips = clips.to(device)
            y_p, y_i, y_w, y_d = y_p.to(device), y_i.to(device), y_w.to(device), y_d.to(device)

            optimizer.zero_grad()
            out_p, out_i, out_w, out_d = model(clips)
            loss = (
                safe_ce(criterion, out_p, y_p)
                + safe_ce(criterion, out_i, y_i)
                + safe_ce(criterion, out_w, y_w)
                + 0.5 * safe_ce(criterion, out_d, y_d)
            )
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * clips.size(0)
            correct_dec += (out_d.argmax(1) == y_d).sum().item()
            n += clips.size(0)

        print(f"Epoch {epoch+1}/{EPOCHS}  loss={total_loss/n:.4f}  decision_acc={correct_dec/n:.3f}")

    OUT_PATH.parent.mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), OUT_PATH)
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
