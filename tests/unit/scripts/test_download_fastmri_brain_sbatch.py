"""Unit tests for scripts/data/download_fastmri_brain.sbatch.

A local server that honours ``Range`` stands in for S3, and a fake ``sbatch``
records what the planner submits. Each guard is planted: a resume whose local
tail disagrees with the remote bytes, a selection larger than the free space, an
expired link, an unknown archive name; and the links' query strings, which are
the credential, must not reach the ``.done`` record.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import tarfile
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import urlsplit

import pytest

SBATCH = Path(__file__).resolve().parents[3] / "scripts/data/download_fastmri_brain.sbatch"

pytestmark = pytest.mark.skipif(
    shutil.which("curl") is None or shutil.which("xz") is None, reason="needs curl and xz"
)


class _RangeHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        url = urlsplit(self.path)
        name = url.path.rsplit("/", 1)[-1]
        rng = self.headers.get("Range")
        self.server.requests.append((name, rng))  # type: ignore[attr-defined]
        path = self.server.root / name  # type: ignore[attr-defined]
        if "expired" in url.query or not path.exists():
            self.send_error(403)
            return
        data = path.read_bytes()
        start, end = 0, len(data) - 1
        if rng:
            first, _, last = rng.removeprefix("bytes=").partition("-")
            start, end = int(first), min(int(last) if last else end, end)
            if start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{len(data)}")
                self.end_headers()
                return
        self.send_response(206 if rng else 200)
        if rng:
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("ETag", '"d41d8cd9-7"')
        self.send_header("Last-Modified", "Mon, 05 Dec 2022 00:51:25 GMT")
        self.end_headers()
        self.wfile.write(data[start : end + 1])

    def log_message(self, *_args: object) -> None:
        pass


@pytest.fixture
def s3(tmp_path: Path) -> Iterator[ThreadingHTTPServer]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
    server.root = tmp_path / "s3"  # type: ignore[attr-defined]
    server.root.mkdir()  # type: ignore[attr-defined]
    server.requests = []  # type: ignore[attr-defined]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server
    server.shutdown()


def _remote(s3: ThreadingHTTPServer, name: str) -> bytes:
    buf = BytesIO()
    with tarfile.open(fileobj=buf, mode="w:xz") as tar:
        payload = os.urandom(300_000)
        info = tarfile.TarInfo(f"multicoil_train/{name}.h5")
        info.size = len(payload)
        tar.addfile(info, BytesIO(payload))
    (s3.root / name).write_bytes(buf.getvalue())  # type: ignore[attr-defined]
    return buf.getvalue()


def _link(s3: ThreadingHTTPServer, name: str, signature: str = "iT%2Fx") -> str:
    host, port = s3.server_address[:2]
    return f"http://{host}:{port}/v2.0/{name}?AWSAccessKeyId=AK&Signature={signature}&Expires=1"


def _url_file(tmp_path: Path, links: list[str]) -> Path:
    private = tmp_path / "private"
    private.mkdir(exist_ok=True)
    lines = [f'curl -C - "{u}" --output {urlsplit(u).path.rsplit("/", 1)[-1]}' for u in links]
    (private / "urls.txt").write_text("\n".join(lines) + "\n")
    return private / "urls.txt"


def _run(tmp_path: Path, *args: str, floor_gb: int = 0, task: int | None = None):
    repo, fake = tmp_path / "repo", tmp_path / "bin"
    (repo / "databases/brain/fastmri").mkdir(parents=True, exist_ok=True)
    fake.mkdir(exist_ok=True)
    (fake / "sbatch").write_text(
        '#!/bin/bash\necho "$*" >> "$SBATCH_LOG"\necho $((1000 + $(wc -l < "$SBATCH_LOG")))\n'
    )
    (fake / "sbatch").chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{fake}:{os.environ['PATH']}",
        "SBATCH_LOG": str(tmp_path / "sbatch.log"),
        "SLURM_SUBMIT_DIR": str(repo),
        "FLOOR_GB": str(floor_gb),
        "ATTEMPTS": "2",
        "RETRY_SLEEP": "0",
    }
    if task is not None:
        env["SLURM_ARRAY_TASK_ID"] = str(task)
    return subprocess.run(
        ["bash", str(SBATCH), *args], env=env, capture_output=True, text=True, check=False
    )


def _local(tmp_path: Path, name: str) -> Path:
    return tmp_path / "repo/databases/brain/fastmri" / name


def _submitted(tmp_path: Path) -> list[str]:
    log = tmp_path / "sbatch.log"
    return log.read_text().splitlines() if log.exists() else []


def test_the_plan_submits_pending_archives_and_a_dependent_extraction(tmp_path, s3) -> None:
    done, pending = "brain_multicoil_train_batch_0.tar.xz", "brain_multicoil_val_batch_0.tar.xz"
    body = _remote(s3, done)
    _remote(s3, pending)
    _local(tmp_path, done).parent.mkdir(parents=True)
    _local(tmp_path, done).write_bytes(body)
    Path(f"{_local(tmp_path, done)}.done").write_text(f"bytes\t{len(body)}\n")
    urls = _url_file(tmp_path, [_link(s3, done), _link(s3, pending)])
    result = _run(tmp_path, str(urls), "--parallel", "3", "--", "--time=48:00:00")
    assert result.returncode == 0, result.stdout + result.stderr
    download, extract = _submitted(tmp_path)
    assert download.startswith("--parsable --array=0-0%3 --time=48:00:00 ")
    assert download.endswith(".selected")
    assert extract == (
        "--parsable --dependency=afterany:1001 scripts/data/extract_fastmri_brain.sbatch "
        f"{done} {pending}"
    )
    (selection,) = urls.parent.glob("urls.txt.*.selected")
    assert stat.S_IMODE(selection.stat().st_mode) == 0o600
    assert selection.read_text() == f"{pending} {_link(s3, pending)}\n"


def test_a_selection_larger_than_the_free_space_is_refused(tmp_path, s3) -> None:
    _remote(s3, "brain_multicoil_train_batch_1.tar.xz")
    urls = _url_file(tmp_path, [_link(s3, "brain_multicoil_train_batch_1.tar.xz")])
    result = _run(tmp_path, str(urls), floor_gb=10**9)
    assert result.returncode == 1
    assert "does not fit" in result.stdout
    assert _submitted(tmp_path) == []
    assert sorted(p.name for p in urls.parent.iterdir()) == ["urls.txt"]


def test_only_narrows_the_selection_and_plan_submits_nothing(tmp_path, s3) -> None:
    names = ["brain_multicoil_train_batch_2.tar.xz", "brain_multicoil_val_batch_1.tar.xz"]
    for name in names:
        _remote(s3, name)
    urls = _url_file(tmp_path, [_link(s3, n) for n in names])
    result = _run(tmp_path, str(urls), "--only", "val", "--plan")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "plan only" in result.stdout
    assert names[1] in result.stdout and names[0] not in result.stdout
    assert _submitted(tmp_path) == []


def test_an_expired_link_and_an_unknown_name_stop_the_plan(tmp_path, s3) -> None:
    name = "brain_multicoil_test_batch_0.tar.xz"
    _remote(s3, name)
    expired = _run(tmp_path, str(_url_file(tmp_path, [_link(s3, name, "expired")])))
    assert expired.returncode == 1
    assert f"{name}: HTTP 403" in expired.stdout
    unknown = _run(tmp_path, str(_url_file(tmp_path, [_link(s3, "knee_x.tar.xz")])))
    assert unknown.returncode == 1
    assert "knee_x.tar.xz is not a fastMRI brain archive name" in unknown.stdout


def _selection(tmp_path: Path, s3: ThreadingHTTPServer, name: str) -> Path:
    sel = tmp_path / "private/sel"
    sel.parent.mkdir(exist_ok=True)
    sel.write_text(f"{name} {_link(s3, name)}\n")
    return sel


def test_a_task_downloads_and_records_no_credential(tmp_path, s3) -> None:
    name = "brain_multicoil_train_batch_3.tar.xz"
    body = _remote(s3, name)
    result = _run(tmp_path, "--task", str(_selection(tmp_path, s3, name)), task=0)
    assert result.returncode == 0, result.stdout + result.stderr
    assert _local(tmp_path, name).read_bytes() == body
    record = Path(f"{_local(tmp_path, name)}.done").read_text()
    assert f"bytes\t{len(body)}" in record and 'etag\t"d41d8cd9-7"' in record
    assert "Signature" not in record and "AWSAccessKeyId" not in record


def test_a_task_resumes_a_partial_whose_tail_matches(tmp_path, s3) -> None:
    name = "brain_multicoil_train_batch_4.tar.xz"
    body = _remote(s3, name)
    have = len(body) * 3 // 5
    _local(tmp_path, name).parent.mkdir(parents=True)
    _local(tmp_path, name).write_bytes(body[:have])
    result = _run(tmp_path, "--task", str(_selection(tmp_path, s3, name)), task=0)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"resuming at {have}" in result.stdout
    assert _local(tmp_path, name).read_bytes() == body
    assert (name, f"bytes={have}-") in s3.requests  # type: ignore[attr-defined]


def test_a_task_refuses_to_append_to_a_foreign_tail(tmp_path, s3) -> None:
    name = "brain_multicoil_train_batch_5.tar.xz"
    body = _remote(s3, name)
    foreign = body[: len(body) // 2 - 1] + bytes([body[len(body) // 2 - 1] ^ 0xFF])
    _local(tmp_path, name).parent.mkdir(parents=True)
    _local(tmp_path, name).write_bytes(foreign)
    result = _run(tmp_path, "--task", str(_selection(tmp_path, s3, name)), task=0)
    assert result.returncode == 1
    assert "differ from the local tail" in result.stdout
    assert _local(tmp_path, name).read_bytes() == foreign
    assert not Path(f"{_local(tmp_path, name)}.done").exists()


def test_the_header_names_the_account_not_a_partition_and_keeps_the_floor() -> None:
    text = SBATCH.read_text()
    assert "#SBATCH --partition" not in text and "#SBATCH --account=johnsson" in text
    assert "FLOOR_GB=${FLOOR_GB:-400}" in text
    assert "--dependency=afterany" in text
