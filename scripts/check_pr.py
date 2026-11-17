"""Gate for pull requests: a contributor PR may only add result files.

Runs from the base branch (pull_request_target), so a PR cannot change the
rules it is checked against. It only reads the PR's files as data through
git; nothing from the PR is executed.

Rules for every PR:
  - no file over MAX_BYTES, no symlinks or submodules.
Rules for a PR from someone outside MAINTAINERS:
  - only new files, at most MAX_FILES, at one of
      results/<profile>/<run_id>.json
      results/<profile>/<modality>/<run_id>.json
      results/<profile>/trials/<run_id>.json
  - existing results can't be changed or deleted;
  - each file is a JSON object whose run_id matches its file name;
  - a trials file needs its result, results/<profile>/<run_id>.json.
Changed result files are copied to --out, keeping their paths, so that
`lemonmatrix validate-submission` can check them against the schemas.

Usage: check_pr.py --base SHA --head REF --author LOGIN --maintainers a,b --out DIR
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

MAX_BYTES = 1_000_000
MAX_FILES = 200
MODALITIES = ("classify", "tts", "stt", "imagegen", "audiogen", "embeddings", "rerank", "meshgen", "omni")
UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
PROFILE = r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}"
RESULT = re.compile(rf"^results/(?P<profile>{PROFILE})/(?:(?P<sub>{'|'.join(MODALITIES)}|trials)/)?(?P<id>{UUID})\.json$")


def git(*args: str) -> bytes:
    return subprocess.run(["git", *args], check=True, capture_output=True).stdout


def changes(base: str, head: str):
    """(status, mode, path) for each file the PR changes, from the merge base."""
    raw = git("diff", "--raw", "--no-renames", "--no-abbrev", "-z", f"{base}...{head}").split(b"\0")
    out = []
    for meta, path in zip(raw[0::2], raw[1::2]):
        if not meta:
            continue
        fields = meta.decode().lstrip(":").split()
        out.append((fields[4], fields[1], path.decode("utf-8", "replace")))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--author", required=True)
    ap.add_argument("--maintainers", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    maintainer = a.author.lower() in {m.strip().lower() for m in a.maintainers.split(",") if m.strip()}
    errors: list[tuple[str, str]] = []
    copied = 0
    files = changes(a.base, a.head)

    if not maintainer and len(files) > MAX_FILES:
        errors.append(("", f"{len(files)} files changed; a result PR has at most {MAX_FILES}."))

    for status, mode, path in files:
        if status == "D":
            if not maintainer:
                errors.append((path, "deletes a file; only maintainers can remove files."))
            continue
        if mode in ("120000", "160000"):
            errors.append((path, "symlinks and submodules are not accepted."))
            continue
        size = int(git("cat-file", "-s", f"{a.head}:{path}"))
        if size > MAX_BYTES:
            errors.append((path, f"{size / 1e6:.1f} MB; files over {MAX_BYTES / 1e6:.0f} MB are not accepted. "
                                 "Model weights and checkpoints belong on a model hub, not here."))
            continue

        m = RESULT.match(path)
        if not maintainer:
            if not m:
                errors.append((path, "only result files can be added: results/<profile>/[<modality>|trials/]<run_id>.json, "
                                     "as written by `lemonmatrix submit`."))
                continue
            if status != "A":
                errors.append((path, "changes an existing result; results can only be added."))
                continue
            if mode != "100644":
                errors.append((path, f"file mode {mode}; results are plain files (100644)."))
                continue
        if not m:
            continue

        blob = git("show", f"{a.head}:{path}")
        try:
            data = json.loads(blob.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append((path, f"not valid UTF-8 JSON: {exc}"))
            continue
        if not isinstance(data, dict) or data.get("run_id") != m["id"]:
            errors.append((path, "run_id inside the file must match the file name."))
            continue
        if m["sub"] == "trials":
            if not isinstance(data.get("trials"), list):
                errors.append((path, "a trials file needs a \"trials\" list."))
                continue
            try:
                git("cat-file", "-e", f"{a.head}:results/{m['profile']}/{m['id']}.json")
            except subprocess.CalledProcessError:
                errors.append((path, f"no matching result results/{m['profile']}/{m['id']}.json."))
                continue

        target = Path(a.out) / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blob)
        copied += 1

    role = "maintainer" if maintainer else "contributor"
    summary = [f"### PR guard ({role} PR, {len(files)} file(s) changed)", ""]
    for path, msg in errors:
        print(f"::error{' file=' + path if path else ''}::{msg}")
        summary.append(f"- ❌ `{path or 'PR'}`: {msg}")
    if not errors:
        summary.append(f"- ✅ {copied} result file(s) passed the path and format checks.")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as fh:
            fh.write("\n".join(summary) + "\n")
    print("\n".join(summary))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
