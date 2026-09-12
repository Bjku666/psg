import zipfile
from pathlib import Path

from models.gssr_p0b_v1.prepare_panoptic_gt import safe_nested_members, selected_members


def test_only_val_panoptic_members_are_selected(tmp_path: Path):
    path = tmp_path / "sample.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("annotations/panoptic_val2017.json", "{}")
        archive.writestr("annotations/panoptic_val2017.zip", b"zip")
        archive.writestr("annotations/panoptic_train2017/x.png", b"png")
    with zipfile.ZipFile(path) as archive:
        names = [info.filename for info in selected_members(archive)]
    assert names == [
        "annotations/panoptic_val2017.json",
        "annotations/panoptic_val2017.zip",
    ]


def test_nested_val_prefix_is_safe(tmp_path: Path):
    path = tmp_path / "nested.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("panoptic_val2017/x.png", b"png")
    with zipfile.ZipFile(path) as archive:
        assert [info.filename for info in safe_nested_members(archive)] == [
            "panoptic_val2017/x.png"
        ]
