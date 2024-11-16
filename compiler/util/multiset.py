from collections import Counter


class Multiset:
    """
    In mathematics, a multiset (or bag, or mset) is a modification of the concept of a
    set that, unlike a set, allows for multiple instances for each of its elements.
    The number of instances given for each element is called the multiplicity of
    that element in the multiset.
    """

    def __init__(self, iterable=None):
        """Initialize the multiset with an optional iterable"""
        self.counter = Counter(iterable) if iterable is not None else Counter()

    def add(self, item, count=1):
        """Add an item to the multiset with a specified count (default is 1)"""
        self.counter[item] += count

    def remove(self, item, count=1):
        """Remove an item from the multiset by a specified count (default is 1)"""
        if self.counter[item] >= count:
            self.counter[item] -= count
        if self.counter[item] <= 0:
            del self.counter[item]

    def count(self, item):
        """Return the count of an item in the multiset"""
        return self.counter[item]

    def is_subset(self, other):
        """Check if this multiset is a subset of another multiset"""
        for item, count in self.counter.items():
            if other.counter[item] < count:
                return False
        return True

    def union(self, other):
        """Return a new multiset that is the union of this and another multiset"""
        return Multiset(self.counter + other.counter)

    def intersection(self, other):
        """Return a new multiset that is the intersection of this and another multiset"""
        return Multiset(self.counter & other.counter)

    def difference(self, other):
        """Return a new multiset that is the difference of this and another multiset"""
        return Multiset(self.counter - other.counter)

    def __contains__(self, item):
        """Check if an element is in the multiset"""
        return item in self.counter

    def __iter__(self):
        """Allow iteration over the multiset as (element, cardinality) pairs"""
        return iter(self.counter.items())

    def __repr__(self):
        """String representation of the multiset"""
        return f"Multiset({self.counter})"
