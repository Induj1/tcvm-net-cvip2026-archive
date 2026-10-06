# TCVM-Net Recovery and Handoff

## Final Frozen State

Title: **TCVM-Net: Temporal Consistency Verification for Robust Traffic Surveillance Under Physical-Style Adversarial Attacks**

The preserved manuscript is anonymous, uses Springer LNCS `llncs` v2.26, and compiles to 15 A4 pages. The final public code commit at archive time is `8665bccd22d14ee2d1b9f876bdec040a1433bc2d`.

Important measured results:

- YOLOv8n clean helmet test: 0.628 mAP@50 and 0.415 mAP@50:95.
- Paired PGD evaluation: 0.988 ASR and mAP@50 reduction from 0.675 to 0.229.
- Controlled reflective probe: TCVM-Net improves mAP@50 from 0.838 to 0.944 with 0.958 anomaly accuracy.
- Public 30-clip HELMET benchmark: 0.887 held-out anomaly F1 at 0.021 FPR.
- Twenty fixed clip-disjoint resplits: median F1 0.889 and median FPR 0.020.
- Frozen reflective transfer: TCVM-Net 0.887/0.021 F1/FPR versus reference recovery 0.839/0.012.
- Quarter-resolution flow: 14.10 FPS on the calibrated benchmark.
- Detector-specific patch: 0.406 held-out ASR but only 0.103 anomaly F1, documenting a temporally smooth failure case.

The claim boundary is deliberate: these are physical-style digital perturbations on real images/video, not a printed physical-world validation. TCVM-Net is an empirical verification layer, not a certified defense.

## Verify the Code

From `code/`:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
python -m pytest -q
python -m compileall -q src scripts tests tools
```

Expected unit-test result at archive time: `11 passed`.

## Compile the Manuscript

From `manuscript/` with TeX Live and the bundled LNCS class/style:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Expected output: `main.pdf`, 15 A4 pages, approximately 1.56 MB, with no unresolved citations, references, or overfull boxes.

## Restore the Datasets

Kaggle credentials must remain outside the repository.

```powershell
python -m kaggle datasets download -d andrewmvd/hard-hat-detection -p data/source/helmet --unzip
python -m kaggle datasets download -d a7madmostafa/bdd100k-yolo -p data/source/bdd100k-yolo-kaggle --unzip
python -m kaggle datasets download -d ayushraj2349/sample-videos-for-helmet-detection-on-yolov8 -p data/source/helmet_videos --unzip
python -m kaggle datasets download -d hozngvan/helmet-detection -p data/source/helmet-detection-kaggle --unzip
```

Dataset preparation and exact experiment commands are documented in `code/README.md` and `code/docs/quick_start.md`.

## Preserved Checkpoints

- `weights/helmet_yolov8n_best.pt`: clean YOLOv8n.
- `weights/helmet_yolov8s_best.pt`: clean YOLOv8s.
- `weights/helmet_yolov8n_advtrain_best.pt`: adversarially augmented YOLOv8n.

These are the final `best.pt` checkpoints. Intermediate epoch checkpoints are omitted because they are regenerable and add more than 700 MB.

## Integrity Anchors

- Final anonymous PDF SHA-256: `005582A6C6DD6A85629E347F95B5297C11383122B414F80B28977AB1F79C63E0`
- Anonymous LNCS source ZIP SHA-256: `1BD8E28B9211917DA7C36E6ABABC2AC9CEE9A497154173F0FC763ED58AAA4D3F`
- YOLOv8n best checkpoint: `E43D687D7E1E150ECE418FBCE1AFF39B69DF8782F5967EF1AE05B29AB35E7023`
- YOLOv8s best checkpoint: `E6DDD5A23E97A169B4A4CF4A06A1F077B8B8F6E0478824A2D4A15324F30C38A5`
- AdvAug YOLOv8n best checkpoint: `F1795072E091DC0A46DC4FA3FC527E5DA6510146A61B24F1331CB18A4EACD878`

Use `SHA256SUMS.csv` to verify every individual archived file.

## Presentation

Open `code/README.md` for the visual repository overview and `code/docs/presentation_walkthrough.md` for a five-to-seven-minute presentation sequence. The architecture, attack grid, temporal confidence plot, Grad-CAM comparison, robustness plot, ablation chart, and speed/robustness plot are under `code/outputs/figures/`.

## Submission Safety

- For anonymous review, use the preserved anonymous PDF and do not cite the public GitHub URL if the review policy prohibits identity-revealing links.
- For camera-ready submission, restore the author block from `AUTHOR_METADATA.md` and follow the conference's current instructions.
- Do not claim printed physical validation, universal robustness, certified robustness, or full ADAV/Themis reproduction.
- Do not replace generated metrics without rerunning the corresponding experiment and retaining the new logs.
