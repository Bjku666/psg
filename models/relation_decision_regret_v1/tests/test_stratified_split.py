from models.relation_decision_regret_v1.stratified_split import make_stratified_split, predicate_coverage


def test_stratified_split_keeps_groups_and_predicate_coverage():
    rows = []
    for image_id in range(12):
        rows.append({"image_id": image_id, "file_name": f"physical-{image_id // 2}.jpg",
                     "relations": [[0, 1, image_id % 3]]})
    rows.append({"image_id": 99, "file_name": "test.jpg", "relations": [[0, 1, 0]]})
    psg = {"data": rows, "test_image_ids": [99], "predicate_classes": ["a", "b", "c"]}
    split = make_stratified_split(psg, seed=3)
    assert not (set(split["fit"]) & set(split["dev"]))
    assert not (set(split["fit"]) & set(split["confirm"]))
    assert not (set(split["dev"]) & set(split["confirm"]))
    assert "99" not in set().union(*(set(v) for v in split.values()))
    assert predicate_coverage(psg, split["dev"])
    assert predicate_coverage(psg, split["confirm"])
