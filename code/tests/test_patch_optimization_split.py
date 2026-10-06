from pathlib import Path

from scripts.attacks.optimize_yolov8_patch import select_clip_disjoint_optimization_paths


def test_clip_disjoint_patch_selection_round_robins_clips() -> None:
    paths = [
        Path(f"images/clip{clip_id:02d}_frame_{frame_id:04d}.jpg")
        for clip_id in (1, 3, 7)
        for frame_id in range(4)
    ]

    selected = select_clip_disjoint_optimization_paths(paths, [1, 3, 7], limit=6)

    assert [path.stem for path in selected] == [
        "clip01_frame_0000",
        "clip03_frame_0000",
        "clip07_frame_0000",
        "clip01_frame_0001",
        "clip03_frame_0001",
        "clip07_frame_0001",
    ]
