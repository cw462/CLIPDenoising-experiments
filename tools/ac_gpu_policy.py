"""Measured conservative admission for three models sharing one GPU worker."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import psutil


POLICY = {
    "schema": "ac_triplet_gpu_admission_v1",
    "sample_gap_seconds": 3.0,
    "nvidia_query_timeout_seconds": 10,
    "minimum_remaining_free_mib": 8192,
    "new_job_reservation_mib": {"train": 8192, "eval": 6144, "check": 8192},
    "maximum_own_jobs_with_external": 2,
    "maximum_own_jobs_without_external": {"train": 1, "eval": 2, "check": 1},
    "unallocated_running_jobs": "subtract their full per-kind reservation from observed free memory",
    "eligibility": "both independent samples must satisfy all limits; use conservative sample values",
    "query_failure": "do not admit GPU jobs; leave existing jobs untouched",
    "external_jobs": "observe only; never signal, reprioritize or modify",
}


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def nvidia_rows(query):
    output = subprocess.check_output(
        ["nvidia-smi", query, "--format=csv,noheader,nounits"],
        text=True, stderr=subprocess.PIPE, timeout=POLICY["nvidia_query_timeout_seconds"])
    return list(csv.reader(line for line in output.splitlines() if line.strip()))


def gpu_snapshot(queue):
    """Read capacity and classify only processes whose creation time matches."""
    selected = set(queue.a.get("gpu_indices", [0, 1]))
    devices = {}
    for fields in nvidia_rows("--query-gpu=index,uuid,memory.free"):
        if len(fields) != 3:
            raise RuntimeError("Malformed GPU capacity response")
        index, uuid, free = [field.strip() for field in fields]
        index, free = int(index), float(free)
        if not math.isfinite(free) or free < 0:
            raise RuntimeError("GPU free memory is unavailable or invalid")
        if index in selected:
            if uuid in devices or any(value["index"] == index for value in devices.values()):
                raise RuntimeError("Duplicate GPU identity in capacity response")
            devices[uuid] = {"index": index, "free_mib": free, "external": [],
                             "own_count": 0, "unallocated_reservation_mib": 0}
    if not devices:
        raise RuntimeError("No configured GPU returned by nvidia-smi")
    own_pids, trees = {os.getpid()}, []
    for job in queue.state["jobs"].values():
        if job.get("status") != "running" or job.get("gpu") is None:
            continue
        identity = job.get("process", {})
        pids = set()
        try:
            if identity.get("host") != socket.gethostname():
                raise RuntimeError("Running GPU job has a foreign or absent process identity")
            process = psutil.Process(identity["pid"])
            if abs(process.create_time() - identity["create_time"]) < 0.1:
                pids = {process.pid, *(child.pid for child in process.children(recursive=True))}
        except psutil.NoSuchProcess:
            # The original queue reconciles exit files; retain its slot meanwhile.
            pass
        own_pids.update(pids)
        trees.append((job, pids))
    visible = {}
    for fields in nvidia_rows("--query-compute-apps=gpu_uuid,pid"):
        if len(fields) != 2:
            raise RuntimeError("Malformed GPU process response")
        uuid, pid = [field.strip() for field in fields]
        pid = int(pid)
        if uuid in devices:
            visible.setdefault(uuid, set()).add(pid)
            if pid not in own_pids:
                devices[uuid]["external"].append(pid)
    for device_uuid, device in devices.items():
        for job, pids in trees:
            if job.get("gpu") == device["index"]:
                device["own_count"] += 1
                if not (pids & visible.get(device_uuid, set())):
                    kind = job.get("kind", "train")
                    device["unallocated_reservation_mib"] += POLICY["new_job_reservation_mib"].get(kind, 8192)
        device["external"] = sorted(set(device["external"]))
    return {value["index"]: value for value in devices.values()}


def admission_options(first, second, kind, configured_training_limit=4):
    """Pure decision function; both samples must independently be admissible."""
    if kind not in POLICY["new_job_reservation_mib"]:
        raise ValueError("Unsupported GPU job kind: " + str(kind))
    options = []
    required = POLICY["new_job_reservation_mib"][kind]
    ordinary_limit = POLICY["maximum_own_jobs_without_external"][kind]
    if kind == "train":
        ordinary_limit = min(ordinary_limit, int(configured_training_limit))
    for index in sorted(set(first) & set(second)):
        samples = (first[index], second[index])
        external = any(value["external"] for value in samples)
        maximum = min(ordinary_limit, POLICY["maximum_own_jobs_with_external"]) if external else ordinary_limit
        if all(value["own_count"] < maximum
               and value["free_mib"] - value["unallocated_reservation_mib"] - required
               >= POLICY["minimum_remaining_free_mib"] for value in samples):
            options.append((max(value["own_count"] for value in samples),
                            -min(value["free_mib"] - value["unallocated_reservation_mib"] for value in samples),
                            index))
    return sorted(options)


def audit_decision(queue, value):
    event = {"timestamp": utc_now(), **value}
    with queue.shared_resource_log.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()


def shared_gpu_options(queue, kind):
    try:
        first = gpu_snapshot(queue)
        # A second sample is useful only when the first can admit something.
        if not admission_options(first, first, kind, queue.a.get("jobs_per_gpu", 4)):
            audit_decision(queue, {"kind": kind, "status": "waiting", "first": first, "choices": []})
            return []
        time.sleep(POLICY["sample_gap_seconds"])
        second = gpu_snapshot(queue)
        choices = admission_options(first, second, kind, queue.a.get("jobs_per_gpu", 4))
        audit_decision(queue, {"kind": kind, "status": "admit" if choices else "waiting",
                               "first": first, "second": second, "choices": choices})
        return choices
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, RuntimeError, psutil.Error) as error:
        # Query/identity errors must never look like an empty or free GPU.
        audit_decision(queue, {"kind": kind, "status": "query_error", "choices": [],
                               "error_type": type(error).__name__, "error": str(error)})
        print(f"GPU admission deferred: {type(error).__name__}: {error}", flush=True)
        return []
