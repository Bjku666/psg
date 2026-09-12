from models.gssr_p0b_v1.audit_identity_contract import audit_identity


def test_duplicate_scene_rows_are_grouped_by_physical_filename():
    annotation = [{"bbox": [0, 0, 1, 1], "category_id": 0}]
    psg = {
        "test_image_ids": ["a", "b"],
        "data": [
            {"image_id": "a", "file_name": "val/x.jpg", "relations": [[0, 0, 0]], "annotations": annotation, "pan_seg_file_name": "x.png"},
            {"image_id": "b", "file_name": "val/x.jpg", "relations": [[0, 0, 1]], "annotations": annotation, "pan_seg_file_name": "x.png"},
        ],
    }
    candidates = {"data": [{"image_id": 1, "file_name": "val/x.jpg"}]}
    report = audit_identity(psg, candidates, "test", True)
    assert report["summary"]["scene_rows"] == 2
    assert report["summary"]["unique_filenames"] == 1
    assert report["summary"]["extra_scene_rows_from_duplicate_filenames"] == 1
    assert report["summary"]["bootstrap_sampling_unit"] == "physical_image_file_name"
    assert report["duplicate_groups"][0]["identical_annotations"]
    assert not report["duplicate_groups"][0]["identical_relations"]

