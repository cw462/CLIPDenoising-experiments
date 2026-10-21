"""Triplet memory reservations must account for three resident models."""
from tools.ac_gpu_policy import admission_options


def sample(free=22000, count=0, external=True, pending=0):
    return {0: {"index": 0, "free_mib": free, "own_count": count,
                "external": [123] if external else [], "unallocated_reservation_mib": pending}}


def test_triplet_one_worker_per_gpu_with_eight_gib_reserve():
    assert admission_options(sample(), sample(), "train")
    assert not admission_options(sample(16383), sample(), "train")
    assert not admission_options(sample(count=1), sample(count=1), "train")
    assert not admission_options(sample(pending=8192), sample(pending=8192), "train")


def test_eval_requires_both_samples_and_leaves_external_headroom():
    assert admission_options(sample(20000), sample(20000), "eval")
    assert not admission_options(sample(20000), sample(14000), "eval")
    assert not admission_options(sample(count=2), sample(count=2), "eval")
