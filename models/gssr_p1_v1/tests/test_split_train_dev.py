from models.gssr_p1_v1.split_train_dev import make_split

def test_split_is_grouped_and_excludes_test():
    psg = {"train_image_ids": [1, 2, 3], "test_image_ids": [9], "data": [
        {"image_id": 1, "file_name": "a.jpg"}, {"image_id": 2, "file_name": "a.jpg"},
        {"image_id": 3, "file_name": "b.jpg"}, {"image_id": 9, "file_name": "z.jpg"}]}
    split = make_split(psg, seed=7)
    assert set().union(*map(set, split.values())) == {"1", "2", "3"}
    assert not (set(split["fit"]) & set(split["dev"]))
    assert set(split["fit"]) != {"1"} or set(split["dev"]) != {"2"}

