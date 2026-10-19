from evla.common.stamped_buffer import StampedBuffer, wait_until


def test_get_exact_and_tolerance():
    b = StampedBuffer[str]()
    for t in (0, 100, 200, 300):
        b.add(t, f"m{t}")
    assert b.get(200, 0) == "m200"
    assert b.get(205, 10) == "m200"
    assert b.get(250, 10) is None


def test_out_of_order_insert_and_latest():
    b = StampedBuffer[int]()
    b.add(300, 3)
    b.add(100, 1)
    b.add(200, 2)
    assert b.latest_at_or_before(250) == 2
    assert b.latest_at_or_before(50) is None
    assert b.newest_stamp() == 300


def test_maxlen_drops_oldest():
    b = StampedBuffer[int](maxlen=3)
    for t in range(5):
        b.add(t, t)
    assert b.get(0, 0) is None and b.get(4, 0) == 4


def test_wait_until():
    assert wait_until(lambda: True, 0.1)
    assert not wait_until(lambda: False, 0.01)
