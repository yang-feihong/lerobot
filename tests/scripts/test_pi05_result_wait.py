from threading import Condition, Event, Thread
from types import SimpleNamespace

from lerobot.scripts.pi05_vla_server import AsyncRTCPolicy


def engine():
    value = AsyncRTCPolicy.__new__(AsyncRTCPolicy)
    value._records_lock = Condition()
    value._records = {}
    value._latest_record = None
    value._inference_count = 0
    value._stop = Event()
    value.recorder = SimpleNamespace(action=lambda _: None)
    return value


def test_result_notification_has_no_lost_wakeup():
    policy = engine()
    record = SimpleNamespace(sequence=3)
    received = []
    thread = Thread(target=lambda: received.append(policy.latest_after(2, 20)))
    thread.start()
    policy._publish_record(record)
    thread.join(1)
    assert received == [record]
    assert policy.latest_after(2, 20) is record
    assert policy.latest_after(3, 1) is None


def test_wait_releases_records_lock_and_stop_wakes_waiter():
    policy = engine()
    entered = Event()
    original = policy._records_lock.wait_for

    def waiting(*args, **kwargs):
        entered.set()
        return original(*args, **kwargs)

    policy._records_lock.wait_for = waiting
    result = []
    thread = Thread(target=lambda: result.append(policy.latest_after(0, 1000)))
    thread.start()
    assert entered.wait(1)
    with policy._records_lock:
        policy._stop.set()
        policy._records_lock.notify_all()
    thread.join(1)
    assert result == [None]
