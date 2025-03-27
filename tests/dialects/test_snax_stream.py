from snaxc.dialects.snax_stream import StridePattern


def test_canonicalize_stride_pattern():
    # bounds of 1 are removed
    s = StridePattern([2, 1, 1], [0, 0, 0], [1]).canonicalize()
    assert s == StridePattern([2], [0], [1])

    s = StridePattern([1, 1, 1], [0, 0, 0], [1]).canonicalize()
    assert s == StridePattern([], [], [1])

    # zeros are kept to correctly disable streamers
    s = StridePattern([0, 0, 0], [0, 0, 0], [1]).canonicalize()
    assert s == StridePattern([0, 0, 0], [0, 0, 0], [1])

    # if possible, wrap
    s = StridePattern([4, 4, 4], [1, 4, 16], [1]).canonicalize()
    assert s == StridePattern([4 * 4 * 4], [1], [1])
