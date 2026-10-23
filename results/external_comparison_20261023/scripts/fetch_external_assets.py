"""Download the exact public assets listed in the comparison provenance files."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def entries(model):
    record = json.loads((ROOT / "provenance" / f"{model}.json").read_text(encoding="utf-8"))
    if model == "DnCNN":
        return record["assets"]
    if model == "MaskedDenoising":
        return [item for item in record["files"] if "sha256" in item]
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("all", "DnCNN", "Restormer", "MaskedDenoising"), default="all")
    args = parser.parse_args()
    models = ("DnCNN", "Restormer", "MaskedDenoising") if args.model == "all" else (args.model,)
    for model in models:
        directory = (ROOT / "external" / model).resolve()
        for item in entries(model):
            relative = item.get("file", item.get("path"))
            target = (directory / relative).resolve()
            if directory not in target.parents:
                raise ValueError(f"Invalid asset path: {relative}")
            if target.exists():
                if sha256(target) != item["sha256"]:
                    raise RuntimeError(f"Existing asset differs from the recorded file: {target.name}")
                print(f"Present: {model}/{relative}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".part")
            request = Request(item["url"], headers={"User-Agent": "CLIPDenoising-comparison-assets"})
            with urlopen(request, timeout=90) as source, temporary.open("wb") as destination:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    destination.write(chunk)
            if sha256(temporary) != item["sha256"]:
                raise RuntimeError(f"Downloaded bytes differ from the recorded file: {model}/{relative}")
            temporary.replace(target)
            print(f"Downloaded: {model}/{relative}")


if __name__ == "__main__":
    main()
