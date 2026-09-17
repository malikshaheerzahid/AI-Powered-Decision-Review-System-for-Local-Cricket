"""
Smart DRS - Inference Backend
------------------------------
Serves the trained r3d_18 ball-type classifier (LBW / Legal Balls / No balls / Wide Balls)
to the DRS PRO frontend.

Run with:
    pip install fastapi uvicorn python-multipart torch torchvision opencv-python numpy
    uvicorn app:app --reload --port 8000

Place your trained checkpoint at:  backend/models/best_model.pt
(This must be the COMPLETE file saved by torch.save(model.state_dict(), path) —
 not just the extracted data.pkl / byteorder / version / serialization_id parts.)
"""

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from torchvision.models.video import r3d_18

from rule_engine import lbw_rule_engine

# ---------------- Config (must match training notebook) ----------------
CLASSES = ["LBW", "Legal Balls", "No balls", "Wide Balls"]  # sorted() order used in training
NUM_CLASSES = len(CLASSES)
CLIP_LEN = 16
RESIZE = 112
MEAN = [0.43216, 0.394666, 0.37645]
STD = [0.22803, 0.22145, 0.216989]

CHECKPOINT_PATH = Path(__file__).parent / "models" / "best_model.pt"
LBW_CHECKPOINT_PATH = Path(__file__).parent / "models" / "lbw_submodel.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PITCHING_CLASSES = ["OUTSIDE_LEG", "OUTSIDE_OFF", "IN_LINE"]
IMPACT_CLASSES = ["OUTSIDE", "IN_LINE"]
WICKETS_CLASSES = ["MISSING", "HITTING"]
PRE_GRAPHIC_FRACTION = 0.55  # only look at first 55% of clip (matches training)

# ---------------- Model ----------------
def build_model(num_classes: int) -> nn.Module:
    model = r3d_18(weights=None)  # weights are overwritten by our checkpoint below
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


model = build_model(NUM_CLASSES).to(device)
model_ready = False

if CHECKPOINT_PATH.exists():
    try:
        state_dict = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        model_ready = True
        print(f"Loaded checkpoint from {CHECKPOINT_PATH}")
    except Exception as e:
        print(f"WARNING: failed to load checkpoint ({e}). API will return 503 until fixed.")
else:
    print(f"WARNING: no checkpoint found at {CHECKPOINT_PATH}. API will return 503 until you add it.")


# ---------------- LBW sub-component model (Pitching / Impact / Wickets) ----------------
class MultiHeadLBW(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = r3d_18(weights=None)  # weights overwritten by checkpoint
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head_pitch = nn.Linear(in_features, len(PITCHING_CLASSES))
        self.head_impact = nn.Linear(in_features, len(IMPACT_CLASSES))
        self.head_wkts = nn.Linear(in_features, len(WICKETS_CLASSES))
        self.head_decision = nn.Linear(in_features, 2)  # unused at inference (rule engine decides)

    def forward(self, x):
        feat = self.backbone(x)
        return self.head_pitch(feat), self.head_impact(feat), self.head_wkts(feat)


lbw_model = MultiHeadLBW().to(device)
lbw_model_ready = False

if LBW_CHECKPOINT_PATH.exists():
    try:
        lbw_state = torch.load(LBW_CHECKPOINT_PATH, map_location=device, weights_only=True)
        lbw_model.load_state_dict(lbw_state)
        lbw_model.eval()
        lbw_model_ready = True
        print(f"Loaded LBW sub-model from {LBW_CHECKPOINT_PATH}")
    except Exception as e:
        print(f"WARNING: failed to load LBW sub-model ({e}). LBW breakdown will be unavailable.")
else:
    print(f"NOTE: no LBW sub-model at {LBW_CHECKPOINT_PATH} yet. Run backend/lbw_training/train_lbw.py to create it.")


# ---------------- Preprocessing (mirrors CricketDeliveryDataset from training) ----------------
def sample_frame_indices(total_frames: int, clip_len: int) -> np.ndarray:
    if total_frames <= 0:
        return np.zeros(clip_len, dtype=int)
    if total_frames >= clip_len:
        return np.linspace(0, total_frames - 1, clip_len).astype(int)
    reps = int(np.ceil(clip_len / total_frames))
    return np.tile(np.arange(total_frames), reps)[:clip_len]


def load_clip(path: str, clip_len: int = CLIP_LEN, resize: int = RESIZE) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = set(sample_frame_indices(total, clip_len).tolist())

    frames = {}
    frame_idx = 0
    while cap.isOpened() and len(frames) < len(indices):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (resize, resize))
            frames[frame_idx] = frame
        frame_idx += 1
    cap.release()

    if not frames:
        raise ValueError("Could not read any frames from the uploaded video.")

    ordered_indices = sample_frame_indices(total, clip_len)
    clip = []
    last_valid = np.zeros((resize, resize, 3), dtype=np.uint8)
    for idx in ordered_indices:
        if idx in frames:
            last_valid = frames[idx]
        clip.append(last_valid)
    return np.stack(clip)


def preprocess_clip(clip: np.ndarray) -> torch.Tensor:
    clip = clip.astype(np.float32) / 255.0
    clip = (clip - MEAN) / STD
    clip = np.ascontiguousarray(clip.transpose(3, 0, 1, 2))  # C, T, H, W
    tensor = torch.from_numpy(clip).float().unsqueeze(0)  # add batch dim
    return tensor


def load_clip_pre_graphic(path: str, clip_len: int = CLIP_LEN, resize: int = RESIZE) -> np.ndarray:
    """Same as load_clip, but only samples from the first PRE_GRAPHIC_FRACTION
    of the video -- matches how the LBW sub-model was trained (avoids reading
    the on-screen Pitching/Impact/Wickets/Decision graphic as a shortcut)."""
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    usable = max(int(total * PRE_GRAPHIC_FRACTION), clip_len)
    indices = set(sample_frame_indices(usable, clip_len).tolist())

    frames = {}
    frame_idx = 0
    while cap.isOpened() and len(frames) < len(indices):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (resize, resize))
            frames[frame_idx] = frame
        frame_idx += 1
    cap.release()

    if not frames:
        raise ValueError("Could not read any frames from the uploaded video.")

    ordered_indices = sample_frame_indices(usable, clip_len)
    clip = []
    last_valid = np.zeros((resize, resize, 3), dtype=np.uint8)
    for idx in ordered_indices:
        if idx in frames:
            last_valid = frames[idx]
        clip.append(last_valid)
    return np.stack(clip)


def predict_lbw_breakdown(tmp_path: str) -> dict | None:
    """Runs the LBW sub-model + rule engine. Returns None if the sub-model isn't loaded."""
    if not lbw_model_ready:
        return None
    clip = load_clip_pre_graphic(tmp_path)
    tensor = preprocess_clip(clip).to(device)
    with torch.no_grad():
        out_p, out_i, out_w = lbw_model(tensor)
        p_probs = torch.softmax(out_p, dim=1)[0]
        i_probs = torch.softmax(out_i, dim=1)[0]
        w_probs = torch.softmax(out_w, dim=1)[0]

    pitching = PITCHING_CLASSES[int(torch.argmax(p_probs))]
    impact = IMPACT_CLASSES[int(torch.argmax(i_probs))]
    wickets = WICKETS_CLASSES[int(torch.argmax(w_probs))]

    verdict = lbw_rule_engine(pitching, impact, wickets)
    return {
        "pitching": pitching,
        "impact": impact,
        "wickets": wickets,
        "final_decision": verdict["decision"],
        "reason": verdict["reason"],
        "sub_confidences": {
            "pitching": round(float(p_probs.max()) * 100, 2),
            "impact": round(float(i_probs.max()) * 100, 2),
            "wickets": round(float(w_probs.max()) * 100, 2),
        },
    }


# ---------------- API ----------------
app = FastAPI(title="Smart DRS Inference API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your frontend's origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_ready": model_ready,
        "lbw_submodel_ready": lbw_model_ready,
        "device": str(device),
    }


@app.post("/predict")
async def predict(video: UploadFile = File(...)):
    if not model_ready:
        raise HTTPException(
            status_code=503,
            detail="Model checkpoint not loaded. Add a complete best_model.pt to backend/models/.",
        )

    suffix = Path(video.filename).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await video.read())
        tmp_path = tmp.name

    try:
        clip = load_clip(tmp_path)
        tensor = preprocess_clip(clip).to(device)

        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)[0]

        pred_idx = int(torch.argmax(probs).item())
        result = {
            "predicted_class": CLASSES[pred_idx],
            "confidence": round(float(probs[pred_idx]) * 100, 2),
            "probabilities": {
                cls: round(float(p) * 100, 2) for cls, p in zip(CLASSES, probs.tolist())
            },
        }

        # Always run the LBW sub-model + rule engine for a real
        # Pitching/Impact/Wickets breakdown and OUT/NOT_OUT verdict.
        # (We no longer gate this on the first-stage ball-type classifier --
        # that model was trained on a different-looking dataset than these
        # LBW clips, so its "is this LBW" guess is unreliable here.)
        lbw_breakdown = predict_lbw_breakdown(tmp_path)
        if lbw_breakdown:
            result["lbw_breakdown"] = lbw_breakdown

        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference failed: {e}")
    finally:
        os.unlink(tmp_path)
