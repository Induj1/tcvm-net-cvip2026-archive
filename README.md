# TCVM-Net Private Preservation Archive

This private repository preserves the complete handoff state of the TCVM-Net CVIP 2026 project as of 7 October 2026. It is intended to remain usable even if the original Codex conversation becomes unavailable.

## Contents

- `code/`: complete public research implementation, tests, configurations, selected metrics, and presentation figures.
- `manuscript/`: anonymous 15-page LNCS manuscript, bibliography, generated tables, figures, and compiled PDF.
- `outputs/figures/`: the three external figure files referenced by `manuscript/main.tex`.
- `submission/`: independently compilable anonymous LNCS source ZIP.
- `supplementary/`: supplementary LaTeX source and compiled PDF retained for archival purposes.
- `weights/`: final best YOLOv8n, YOLOv8s, and adversarially trained YOLOv8n checkpoints.
- `RECOVERY.md`: exact recovery, verification, and reproduction instructions.
- `AUTHOR_METADATA.md`: private author and camera-ready metadata.
- `SHA256SUMS.csv`: integrity hashes for every archived file.

Raw public datasets, extracted video frames, caches, and intermediate training epochs are not included. They are regenerable from the documented Kaggle sources. The final best checkpoints and compact experimental evidence are included.

The public code repository remains at <https://github.com/Induj1/tcvm-net-cvip2026>.
