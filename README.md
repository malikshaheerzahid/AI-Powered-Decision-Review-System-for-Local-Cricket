# Smart DRS — Full Integration (Ball-type + Real LBW OUT/NOT-OUT)

## What's in this package
- `index.html` / `script.js` / `style.css` — frontend
- `backend/app.py` — FastAPI server, two chained models:
  1. **Ball-type classifier** (`models/best_model.pt`, already included) → LBW / Legal Balls / No balls / Wide Balls
  2. **LBW sub-model** (`models/lbw_submodel.pt`, **you must train this**) → Pitching / Impact / Wickets, combined via `rule_engine.py` into a real **OUT / NOT OUT** verdict — only runs when the ball-type classifier flags "LBW"
- `backend/rule_engine.py` — hand-coded real LBW law (not a black box):
  - Pitching outside leg → NOT OUT
  - Impact outside off → NOT OUT
  - Wickets missing → NOT OUT
  - Otherwise → OUT
- `backend/lbw_training/` — training script + the 19 labeled clips (`v1-v19.mp4` + `labels.csv`) used to train the LBW sub-model

## ⚠️ One-time step required: train the LBW sub-model
This is **not included pre-trained** (needs your machine's internet access to download
pretrained weights, and takes 15-30 min on CPU). Do this once:

```bash
cd backend/lbw_training
pip install torch torchvision opencv-python numpy
python train_lbw.py
```
This saves `backend/models/lbw_submodel.pt`. Restart `uvicorn` afterwards — you'll see
`Loaded LBW sub-model from ...` in the terminal.

**Until you do this**, LBW appeals will show a generic "flagged for review" message
(same as before) instead of a real OUT/NOT OUT — everything else (Legal/No ball/Wide
ball detection) works immediately without this step.

## Run the backend
```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```
Then open `index.html` in your browser.

## ⚠️ Please read before your presentation — be ready to explain this honestly
1. **The LBW sub-model was trained on only 19 clips** (7 OUT / 12 NOT OUT). This is
   extremely small for a deep learning model — it will not reliably generalize to new
   real-world footage. Treat the demo as a **proof-of-concept that the pipeline works
   end-to-end** (video → ball-type → pitching/impact/wickets → real cricket-law verdict),
   not as a validated, production-accurate system. If asked in your viva, say this
   plainly — it's the honest and defensible answer.
2. **Labels came from OCR'd on-screen graphics** already burned into the training clips
   (the Pitching/Impact/Wickets/Decision boxes visible in the footage), cross-checked
   by eye where OCR was unclear. One clip (`v20`) had no readable overlay and was
   excluded entirely.
3. **To prevent the model from "cheating"** by reading that same on-screen text, both
   training and inference only look at the **first 55% of each clip** (run-up through
   impact, before the graphic appears on screen).
4. The final OUT/NOT-OUT call is **not** learned end-to-end — it's computed by the
   explicit rule engine from the three sub-model predictions. This is intentional:
   it mirrors real LBW law exactly and is far more explainable/defensible than a
   black-box "decision" head trained on 19 examples.
5. The original ball-type classifier (LBW/Legal/No ball/Wide ball, from `CRIC2.ipynb`)
   has its own separate, earlier-documented limitations (94% test accuracy but on a
   tiny, possibly-biased test set — see earlier discussion).

## Not yet wired up
Live Camera analysis (`analyzeCurrentFrame`) still uses the mock animation.
