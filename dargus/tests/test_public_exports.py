def test_top_level_exports():
    from dargus import DBase, Iris

    assert callable(Iris)
    assert callable(DBase)
