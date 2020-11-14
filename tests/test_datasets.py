from tests.context import datasets

def test_load_khan_train():
    res = datasets.load_khan_train()
    assert res['data'].shape == (64, 306)
    assert len(res['labels']) == 64

def test_load_khan_test():
    res = datasets.load_khan_test()
    assert res['data'].shape == (25, 306)
    assert len(res['labels']) == 25

def test_load_vote_repub():
    res = datasets.load_vote_repub()
    assert res['data'].shape == (50, 31)
    assert len(res['labels']) == 50

def test_load_animals():
    res = datasets.load_animals()
    assert res['data'].shape == (20, 6)
    assert len(res['labels']) == 20