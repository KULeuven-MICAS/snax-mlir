from collections import Counter

from compiler.util.multiset import Multiset


def test_multiset_initialization():
    multiset = Multiset([1, 2, 2, 3])
    assert multiset.counter == Counter([1, 2, 2, 3])


def test_multiset_add():
    multiset = Multiset([1, 2])
    multiset.add(2)
    multiset.add(3, 2)
    assert multiset.counter == Counter([1, 2, 2, 3, 3])


def test_multiset_remove():
    multiset = Multiset([1, 2, 2, 3])
    multiset.remove(2)
    multiset.remove(3)
    assert multiset.counter == Counter([1, 2])


def test_multiset_remove_item_not_present():
    multiset = Multiset([1, 2])
    multiset.remove(3)  # Removing an item that isn't there shouldn't throw an error
    assert multiset.counter == Counter([1, 2])


def test_multiset_count():
    multiset = Multiset([1, 2, 2, 3])
    assert multiset.count(2) == 2
    assert multiset.count(3) == 1


def test_multiset_is_subset():
    multiset1 = Multiset([1, 2, 2, 3])
    multiset2 = Multiset([1, 2, 2, 3, 3, 4])
    assert multiset1.is_subset(multiset2)


def test_multiset_is_not_subset():
    multiset1 = Multiset([1, 2, 2, 3, 5])
    multiset2 = Multiset([1, 2, 2, 3, 3, 4])
    assert not multiset1.is_subset(multiset2)


def test_multiset_union():
    multiset1 = Multiset([1, 2, 2])
    multiset2 = Multiset([2, 3])
    union_set = multiset1.union(multiset2)
    assert union_set.counter == Counter([1, 2, 2, 2, 3])


def test_multiset_intersection():
    multiset1 = Multiset([1, 2, 2, 3])
    multiset2 = Multiset([2, 2, 3, 4])
    intersection_set = multiset1.intersection(multiset2)
    assert intersection_set.counter == Counter([2, 2, 3])


def test_multiset_difference():
    multiset1 = Multiset([1, 2, 2, 3])
    multiset2 = Multiset([2, 3])
    difference_set = multiset1.difference(multiset2)
    assert difference_set.counter == Counter([1, 2])
