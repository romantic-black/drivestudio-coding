#!/usr/bin/env python3
"""
Zip selected AutoDL folders in bounded chunks, upload each archive to Aliyun Drive,
then remove the local zip only after upload succeeds.

Dependency:
    python -m pip install -U aligo

First run of aligo usually asks you to scan a QR code in the terminal.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
import shutil
import socket
import stat
import sys
import time
from typing import Any, Iterable, Iterator, Sequence
import zipfile


DEFAULT_SOURCES = [
    "/root/autodl-tmp/streetforward_assets_waymo",
    "/root/autodl-tmp/streetforward_assets_pt80w_40w",
    "/root/autodl-tmp/streetforward_assets_pt60w_40w",
    "/root/autodl-tmp/streetforward_assets_pt150w",
    "/root/autodl-tmp/streetforward_assets_pt120w",
    "/root/autodl-tmp/outputs",
    "/root/autodl-tmp/nuScenes",
    "/root/autodl-tmp/drivestudio-log",
]


BYTES_PER_GIB = 1024**3
COPY_BUFFER_SIZE = 16 * 1024 * 1024


@dataclass(frozen=True)
class ArchiveEntry:
    path: Path
    arcname: str
    kind: str
    size: int


@dataclass(frozen=True)
class ResumeState:
    uploaded_archives: frozenset[str]
    zip_done_archives: frozenset[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chunk zip AutoDL folders and upload archives to Aliyun Drive."
    )
    parser.add_argument(
        "--source",
        action="append",
        help="Source file/folder to upload. Repeat to override defaults.",
    )
    parser.add_argument(
        "--remote-dir",
        default=None,
        help="Aliyun Drive folder path. Default: /AutoDL-backups/<host>-<timestamp>",
    )
    parser.add_argument(
        "--staging-dir",
        default="/root/autodl-tmp/alidrive_zip_staging",
        help="Local directory used for temporary zip files and logs.",
    )
    parser.add_argument(
        "--max-zip-gb",
        type=float,
        default=80.0,
        help="Target maximum uncompressed payload per zip archive.",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=20.0,
        help="Minimum free space to keep on the staging filesystem.",
    )
    parser.add_argument(
        "--compression",
        choices=("stored", "deflated"),
        default="stored",
        help="Zip compression mode. 'stored' is faster and has predictable size.",
    )
    parser.add_argument(
        "--compresslevel",
        type=int,
        default=1,
        help="Compression level for --compression deflated.",
    )
    parser.add_argument(
        "--upload-retries",
        type=int,
        default=3,
        help="Number of upload attempts per archive.",
    )
    parser.add_argument(
        "--retry-sleep",
        type=float,
        default=30.0,
        help="Seconds to sleep between upload retries.",
    )
    parser.add_argument(
        "--account-name",
        default=None,
        help="Optional aligo account name, passed to Aligo(name=...).",
    )
    parser.add_argument(
        "--aligo-config-dir",
        default=None,
        help="Optional aligo config directory for persisted login tokens.",
    )
    parser.add_argument(
        "--symlink-mode",
        choices=("store", "skip", "follow"),
        default="store",
        help="How to handle symlinks. Default stores symlink metadata, not targets.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Glob pattern to exclude. Matched against archive and absolute paths.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and print planned zip chunks without creating or uploading files.",
    )
    parser.add_argument(
        "--resume-state",
        default=None,
        help="Resume from a previous state jsonl file. Reuses that run's archive names.",
    )
    parser.add_argument(
        "--keep-zips",
        action="store_true",
        help="Do not delete zip archives after successful upload.",
    )
    parser.add_argument(
        "--require-remote-size",
        action="store_true",
        help="Fail and keep the zip if remote size cannot be verified.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with later archives after an error. Failed zips are kept.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional log file path. Default is under --staging-dir.",
    )
    parser.add_argument(
        "--stop-after",
        type=int,
        default=None,
        help="Stop after this many archives. Useful for testing.",
    )
    return parser.parse_args()


def configure_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    quiet_noisy_loggers()


def quiet_noisy_loggers() -> None:
    for name in ("aligo", "urllib3", "requests"):
        logging.getLogger(name).setLevel(logging.WARNING)


def now_stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def safe_slug(value: str) -> str:
    safe = []
    for char in value:
        if char.isalnum() or char in ("-", "_", "."):
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe).strip("._") or "archive"


def gib_to_bytes(value: float) -> int:
    return int(value * BYTES_PER_GIB)


def format_gib(num_bytes: int) -> str:
    return f"{num_bytes / BYTES_PER_GIB:.2f} GiB"


def is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def normalize_remote_path(remote_dir: str) -> str:
    remote_dir = remote_dir.strip()
    if not remote_dir or remote_dir == "/":
        return "/"
    return "/" + "/".join(part for part in remote_dir.split("/") if part)


def remote_parts(remote_dir: str) -> list[str]:
    return [part for part in normalize_remote_path(remote_dir).split("/") if part]


def archive_name(source: Path, batch_stamp: str, part_index: int) -> str:
    return f"{safe_slug(source.name)}.{batch_stamp}.part{part_index:06d}.zip"


def append_state(state_file: Path, event: str, **payload: Any) -> None:
    payload = {
        "event": event,
        "time": dt.datetime.now(dt.timezone.utc).isoformat(),
        **payload,
    }
    with state_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")


def batch_stamp_from_state_file(state_file: Path) -> str | None:
    match = re.search(r"alidrive_zip_upload\.(\d{8}-\d{6})\.jsonl$", state_file.name)
    if not match:
        return None
    return match.group(1)


def load_resume_state(state_file: Path) -> ResumeState:
    uploaded: set[str] = set()
    zip_done: set[str] = set()

    with state_file.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid resume state JSON at line {line_number}: {state_file}") from exc

            archive = record.get("archive")
            event = record.get("event")
            if not archive or not event:
                continue
            if event in {"upload_done", "zip_removed"}:
                uploaded.add(str(archive))
            elif event == "zip_done":
                zip_done.add(str(archive))

    return ResumeState(uploaded_archives=frozenset(uploaded), zip_done_archives=frozenset(zip_done))


def should_exclude(path: Path, arcname: str, patterns: Sequence[str]) -> bool:
    if not patterns:
        return False
    arc = arcname.replace(os.sep, "/")
    absolute = str(path)
    return any(fnmatch.fnmatch(arc, pattern) or fnmatch.fnmatch(absolute, pattern) for pattern in patterns)


def arcname_for(source: Path, path: Path) -> str:
    if path == source:
        return source.name
    return f"{source.name}/{path.relative_to(source).as_posix()}"


def symlink_entry(path: Path, source: Path) -> ArchiveEntry:
    try:
        target = os.readlink(path)
    except OSError as exc:
        raise RuntimeError(f"Cannot read symlink target: {path}: {exc}") from exc
    return ArchiveEntry(path=path, arcname=arcname_for(source, path), kind="symlink", size=len(target.encode("utf-8")))


def file_entry(path: Path, source: Path, follow_symlink: bool) -> ArchiveEntry:
    try:
        st = path.stat() if follow_symlink else path.lstat()
    except OSError as exc:
        raise RuntimeError(f"Cannot stat file: {path}: {exc}") from exc
    return ArchiveEntry(path=path, arcname=arcname_for(source, path), kind="file", size=st.st_size)


def iter_source_entries(
    source: Path,
    symlink_mode: str,
    exclude_patterns: Sequence[str],
) -> Iterator[ArchiveEntry]:
    if source.is_symlink() and symlink_mode != "follow":
        entry = symlink_entry(source, source)
        if not should_exclude(source, entry.arcname, exclude_patterns):
            yield entry
        return

    if source.is_file():
        entry = file_entry(source, source, follow_symlink=symlink_mode == "follow")
        if not should_exclude(source, entry.arcname, exclude_patterns):
            yield entry
        return

    if not source.is_dir():
        raise RuntimeError(f"Source is not a regular file or directory: {source}")

    root_entry = ArchiveEntry(path=source, arcname=source.name, kind="dir", size=0)
    if not should_exclude(source, root_entry.arcname, exclude_patterns):
        yield root_entry

    seen_dirs: set[tuple[int, int]] = set()

    def on_walk_error(exc: OSError) -> None:
        raise RuntimeError(f"Cannot walk source tree: {exc}") from exc

    for current_root, dir_names, file_names in os.walk(
        source,
        topdown=True,
        followlinks=symlink_mode == "follow",
        onerror=on_walk_error,
    ):
        current = Path(current_root)
        try:
            current_st = current.stat()
            seen_dirs.add((current_st.st_dev, current_st.st_ino))
        except OSError as exc:
            raise RuntimeError(f"Cannot stat directory: {current}: {exc}") from exc

        kept_dirs: list[str] = []
        for dirname in sorted(dir_names):
            path = current / dirname
            arcname = arcname_for(source, path)
            if should_exclude(path, arcname, exclude_patterns):
                logging.info("excluded directory: %s", path)
                continue

            if path.is_symlink() and symlink_mode != "follow":
                if symlink_mode == "store":
                    yield symlink_entry(path, source)
                else:
                    logging.warning("skipped symlink directory: %s", path)
                continue

            if symlink_mode == "follow":
                try:
                    st = path.stat()
                except OSError as exc:
                    raise RuntimeError(f"Cannot stat directory: {path}: {exc}") from exc
                key = (st.st_dev, st.st_ino)
                if key in seen_dirs:
                    logging.warning("skipped already-seen directory target: %s", path)
                    continue

            kept_dirs.append(dirname)
            yield ArchiveEntry(path=path, arcname=arcname, kind="dir", size=0)

        dir_names[:] = kept_dirs

        for filename in sorted(file_names):
            path = current / filename
            arcname = arcname_for(source, path)
            if should_exclude(path, arcname, exclude_patterns):
                logging.info("excluded file: %s", path)
                continue

            if path.is_symlink() and symlink_mode != "follow":
                if symlink_mode == "store":
                    yield symlink_entry(path, source)
                else:
                    logging.warning("skipped symlink file: %s", path)
                continue

            yield file_entry(path, source, follow_symlink=symlink_mode == "follow")


def split_entries(
    entries: Iterable[ArchiveEntry],
    max_payload_bytes: int,
) -> Iterator[tuple[list[ArchiveEntry], int]]:
    group: list[ArchiveEntry] = []
    payload_bytes = 0

    for entry in entries:
        if (
            entry.kind == "file"
            and group
            and payload_bytes > 0
            and payload_bytes + entry.size > max_payload_bytes
        ):
            yield group, payload_bytes
            group = []
            payload_bytes = 0

        if entry.kind == "file" and entry.size > max_payload_bytes:
            logging.warning(
                "single file exceeds --max-zip-gb and will be archived alone: %s (%s)",
                entry.path,
                format_gib(entry.size),
            )

        group.append(entry)
        if entry.kind == "file":
            payload_bytes += entry.size

    if group:
        yield group, payload_bytes


def zip_datetime(timestamp: float) -> tuple[int, int, int, int, int, int]:
    value = dt.datetime.fromtimestamp(timestamp)
    if value.year < 1980:
        return (1980, 1, 1, 0, 0, 0)
    return (value.year, value.month, value.day, value.hour, value.minute, value.second)


def add_directory(zf: zipfile.ZipFile, entry: ArchiveEntry) -> None:
    try:
        st = entry.path.lstat()
    except OSError as exc:
        raise RuntimeError(f"Cannot stat directory while zipping: {entry.path}: {exc}") from exc

    info = zipfile.ZipInfo(entry.arcname.rstrip("/") + "/", zip_datetime(st.st_mtime))
    info.create_system = 3
    info.external_attr = ((stat.S_IFDIR | stat.S_IMODE(st.st_mode)) << 16) | 0x10
    zf.writestr(info, b"")


def add_symlink(zf: zipfile.ZipFile, entry: ArchiveEntry) -> None:
    try:
        target = os.readlink(entry.path)
        st = entry.path.lstat()
    except OSError as exc:
        raise RuntimeError(f"Cannot read symlink while zipping: {entry.path}: {exc}") from exc

    info = zipfile.ZipInfo(entry.arcname, zip_datetime(st.st_mtime))
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    zf.writestr(info, target.encode("utf-8"))


def add_file(zf: zipfile.ZipFile, entry: ArchiveEntry, compression: int) -> None:
    try:
        before = entry.path.stat()
    except OSError as exc:
        raise RuntimeError(f"Cannot stat file while zipping: {entry.path}: {exc}") from exc

    info = zipfile.ZipInfo(entry.arcname, zip_datetime(before.st_mtime))
    info.create_system = 3
    info.compress_type = compression
    info.external_attr = (stat.S_IFREG | stat.S_IMODE(before.st_mode)) << 16
    info.file_size = before.st_size

    try:
        with entry.path.open("rb") as source_handle, zf.open(info, "w") as zip_handle:
            shutil.copyfileobj(source_handle, zip_handle, length=COPY_BUFFER_SIZE)
        after = entry.path.stat()
    except OSError as exc:
        raise RuntimeError(f"Cannot add file to zip: {entry.path}: {exc}") from exc

    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
        raise RuntimeError(f"File changed while zipping, refusing archive: {entry.path}")


def build_manifest(
    source: Path,
    entries: Sequence[ArchiveEntry],
    payload_bytes: int,
    archive_path: Path,
    compression_name: str,
) -> dict[str, Any]:
    return {
        "archive_name": archive_path.name,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": str(source),
        "payload_bytes": payload_bytes,
        "compression": compression_name,
        "entries": [
            {
                "path": str(entry.path),
                "arcname": entry.arcname,
                "kind": entry.kind,
                "size": entry.size,
            }
            for entry in entries
        ],
    }


def build_zip(
    archive_path: Path,
    source: Path,
    entries: Sequence[ArchiveEntry],
    payload_bytes: int,
    compression_name: str,
    compresslevel: int,
) -> int:
    if archive_path.exists():
        raise RuntimeError(f"Refusing to overwrite existing archive: {archive_path}")

    temp_path = archive_path.with_name(archive_path.name + ".partial")
    if temp_path.exists():
        temp_path.unlink()

    compression = zipfile.ZIP_STORED if compression_name == "stored" else zipfile.ZIP_DEFLATED
    level = None if compression == zipfile.ZIP_STORED else compresslevel

    logging.info("creating zip: %s (payload %s)", archive_path, format_gib(payload_bytes))
    try:
        with zipfile.ZipFile(
            temp_path,
            mode="w",
            compression=compression,
            allowZip64=True,
            compresslevel=level,
        ) as zf:
            written_dirs: set[str] = set()
            for entry in entries:
                if entry.kind == "dir":
                    arcdir = entry.arcname.rstrip("/") + "/"
                    if arcdir not in written_dirs:
                        add_directory(zf, entry)
                        written_dirs.add(arcdir)
                elif entry.kind == "symlink":
                    add_symlink(zf, entry)
                elif entry.kind == "file":
                    add_file(zf, entry, compression=compression)
                else:
                    raise RuntimeError(f"Unknown entry kind: {entry.kind}")

            manifest = build_manifest(source, entries, payload_bytes, archive_path, compression_name)
            manifest_info = zipfile.ZipInfo("_zip_upload_manifest.json")
            manifest_info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(
                manifest_info,
                json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True).encode("utf-8"),
            )
        temp_path.rename(archive_path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise

    archive_size = archive_path.stat().st_size
    logging.info("zip complete: %s (%s)", archive_path, format_gib(archive_size))
    return archive_size


def check_free_space(staging_dir: Path, estimated_archive_bytes: int, min_free_bytes: int) -> None:
    free_bytes = shutil.disk_usage(staging_dir).free
    overhead = max(256 * 1024 * 1024, int(estimated_archive_bytes * 0.02))
    required = estimated_archive_bytes + overhead + min_free_bytes
    if free_bytes < required:
        raise RuntimeError(
            "Not enough free space on staging filesystem: "
            f"free={format_gib(free_bytes)}, required~={format_gib(required)}"
        )


def import_aligo(config_dir: str | None, account_name: str | None) -> Any:
    try:
        import aligo as aligo_module
        from aligo import Aligo
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'aligo'. Install it with: python -m pip install -U aligo"
        ) from exc
    quiet_noisy_loggers()

    if config_dir:
        set_config_folder = getattr(aligo_module, "set_config_folder", None)
        if set_config_folder is None:
            raise RuntimeError("This aligo version does not expose set_config_folder().")
        set_config_folder(config_dir)

    kwargs = {}
    if account_name:
        kwargs["name"] = account_name
    return Aligo(**kwargs)


def object_get(obj: Any, *names: str) -> Any:
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def list_remote_children(ali: Any, parent_file_id: str) -> list[Any]:
    last_exc: Exception | None = None
    for kwargs in (
        {"parent_file_id": parent_file_id},
        {"file_id": parent_file_id},
        {},
    ):
        if kwargs == {} and parent_file_id != "root":
            continue
        try:
            children = ali.get_file_list(**kwargs)
            return list(children or [])
        except TypeError as exc:
            last_exc = exc
            continue
    if last_exc:
        raise last_exc
    return []


def create_remote_folder(ali: Any, folder_name: str, parent_file_id: str) -> Any:
    call_variants = (
        {"name": folder_name, "parent_file_id": parent_file_id, "check_name_mode": "refuse"},
        {"folder_name": folder_name, "parent_file_id": parent_file_id, "check_name_mode": "refuse"},
        {"name": folder_name, "parent_file_id": parent_file_id},
        {"folder_name": folder_name, "parent_file_id": parent_file_id},
    )
    last_exc: Exception | None = None
    for kwargs in call_variants:
        try:
            return ali.create_folder(**kwargs)
        except TypeError as exc:
            last_exc = exc
    if last_exc:
        raise last_exc
    raise RuntimeError("create_folder failed without an exception")


def ensure_remote_folder(ali: Any, remote_dir: str) -> str:
    parent_id = "root"
    if normalize_remote_path(remote_dir) == "/":
        return parent_id

    for part in remote_parts(remote_dir):
        existing_id = None
        for child in list_remote_children(ali, parent_id):
            child_name = object_get(child, "name", "file_name")
            child_type = object_get(child, "type", "file_type")
            if child_name == part and child_type == "folder":
                existing_id = object_get(child, "file_id", "id")
                break

        if existing_id:
            parent_id = existing_id
            continue

        logging.info("creating remote folder: %s", part)
        created = create_remote_folder(ali, part, parent_id)
        created_id = object_get(created, "file_id", "id")
        if not created_id:
            raise RuntimeError(f"Could not read file_id from created folder response: {created!r}")
        parent_id = created_id

    return parent_id


def find_remote_child_by_name(ali: Any, parent_file_id: str, file_name: str) -> Any | None:
    for child in list_remote_children(ali, parent_file_id):
        if object_get(child, "name", "file_name") == file_name:
            return child
    return None


def upload_file_once(ali: Any, archive_path: Path, parent_file_id: str) -> Any:
    call_variants = (
        {"file_path": str(archive_path), "parent_file_id": parent_file_id, "check_name_mode": "refuse"},
        {"local_file": str(archive_path), "parent_file_id": parent_file_id, "check_name_mode": "refuse"},
        {"file_path": str(archive_path), "parent_file_id": parent_file_id},
        {"local_file": str(archive_path), "parent_file_id": parent_file_id},
    )
    last_exc: Exception | None = None
    for kwargs in call_variants:
        try:
            return ali.upload_file(**kwargs)
        except TypeError as exc:
            last_exc = exc

    try:
        return ali.upload_file(str(archive_path), parent_file_id=parent_file_id)
    except TypeError as exc:
        last_exc = exc

    if last_exc:
        raise last_exc
    raise RuntimeError("upload_file failed without an exception")


def upload_with_retries(
    ali: Any,
    archive_path: Path,
    parent_file_id: str,
    retries: int,
    retry_sleep: float,
) -> Any:
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            logging.info("uploading %s (attempt %d/%d)", archive_path.name, attempt, retries)
            uploaded = upload_file_once(ali, archive_path, parent_file_id)
            logging.info("upload complete: %s", archive_path.name)
            return uploaded
        except Exception as exc:
            last_exc = exc
            logging.exception("upload failed: %s", archive_path.name)
            if attempt < retries:
                time.sleep(retry_sleep)
    raise RuntimeError(f"Upload failed after {retries} attempts: {archive_path}") from last_exc


def fetch_remote_file(ali: Any, file_id: str) -> Any:
    try:
        return ali.get_file(file_id)
    except TypeError:
        return ali.get_file(file_id=file_id)


def verify_remote_size(
    ali: Any,
    uploaded: Any,
    local_size: int,
    require_remote_size: bool,
) -> None:
    remote_size = object_get(uploaded, "size", "file_size")
    file_id = object_get(uploaded, "file_id", "id")
    if remote_size is None and file_id:
        try:
            remote = fetch_remote_file(ali, file_id)
            remote_size = object_get(remote, "size", "file_size")
        except Exception:
            logging.exception("could not fetch uploaded file metadata for size verification")

    if remote_size is None:
        message = "remote size was not available from aligo response"
        if require_remote_size:
            raise RuntimeError(message)
        logging.warning("%s; treating successful upload call as sufficient", message)
        return

    if int(remote_size) != int(local_size):
        raise RuntimeError(f"Remote size mismatch: remote={remote_size}, local={local_size}")

    logging.info("remote size verified: %s", format_gib(local_size))


def validate_sources(sources: Sequence[Path], staging_dir: Path) -> None:
    staging_resolved = staging_dir.resolve()
    for source in sources:
        if not source.exists() and not source.is_symlink():
            raise RuntimeError(f"Source does not exist: {source}")
        resolved_source = source.resolve()
        if is_relative_to(staging_resolved, resolved_source):
            raise RuntimeError(
                f"Staging directory must not be inside a source directory: {staging_dir} inside {source}"
            )


def process_archive(
    *,
    args: argparse.Namespace,
    ali: Any,
    source: Path,
    entries: Sequence[ArchiveEntry],
    payload_bytes: int,
    archive_path: Path,
    remote_parent_id: str,
    state_file: Path,
    min_free_bytes: int,
) -> None:
    check_free_space(archive_path.parent, max(payload_bytes, 1), min_free_bytes)
    append_state(
        state_file,
        "zip_start",
        source=str(source),
        archive=str(archive_path),
        payload_bytes=payload_bytes,
        entry_count=len(entries),
    )
    archive_size = build_zip(
        archive_path=archive_path,
        source=source,
        entries=entries,
        payload_bytes=payload_bytes,
        compression_name=args.compression,
        compresslevel=args.compresslevel,
    )
    append_state(
        state_file,
        "zip_done",
        source=str(source),
        archive=str(archive_path),
        archive_bytes=archive_size,
    )

    uploaded = upload_with_retries(
        ali=ali,
        archive_path=archive_path,
        parent_file_id=remote_parent_id,
        retries=args.upload_retries,
        retry_sleep=args.retry_sleep,
    )
    verify_remote_size(
        ali=ali,
        uploaded=uploaded,
        local_size=archive_size,
        require_remote_size=args.require_remote_size,
    )
    append_state(
        state_file,
        "upload_done",
        source=str(source),
        archive=str(archive_path),
        archive_bytes=archive_size,
        remote_file_id=object_get(uploaded, "file_id", "id"),
    )

    if args.keep_zips:
        logging.info("keeping local zip due to --keep-zips: %s", archive_path)
        return

    archive_path.unlink()
    append_state(state_file, "zip_removed", source=str(source), archive=str(archive_path))
    logging.info("removed local zip: %s", archive_path)


def process_existing_archive(
    *,
    args: argparse.Namespace,
    ali: Any,
    source: Path,
    archive_path: Path,
    remote_parent_id: str,
    state_file: Path,
) -> None:
    archive_size = archive_path.stat().st_size
    append_state(
        state_file,
        "resume_upload_existing",
        source=str(source),
        archive=str(archive_path),
        archive_bytes=archive_size,
    )
    logging.info("resuming existing zip upload: %s (%s)", archive_path, format_gib(archive_size))

    existing_remote = find_remote_child_by_name(ali, remote_parent_id, archive_path.name)
    if existing_remote is not None:
        remote_size = object_get(existing_remote, "size", "file_size")
        if remote_size is not None and int(remote_size) == int(archive_size):
            logging.info("remote file already exists with matching size: %s", archive_path.name)
            append_state(
                state_file,
                "upload_done",
                source=str(source),
                archive=str(archive_path),
                archive_bytes=archive_size,
                remote_file_id=object_get(existing_remote, "file_id", "id"),
            )
            if args.keep_zips:
                logging.info("keeping local zip due to --keep-zips: %s", archive_path)
                return
            archive_path.unlink()
            append_state(state_file, "zip_removed", source=str(source), archive=str(archive_path))
            logging.info("removed local zip: %s", archive_path)
            return
        raise RuntimeError(
            f"Remote file already exists with a different or unknown size: {archive_path.name}"
        )

    uploaded = upload_with_retries(
        ali=ali,
        archive_path=archive_path,
        parent_file_id=remote_parent_id,
        retries=args.upload_retries,
        retry_sleep=args.retry_sleep,
    )
    verify_remote_size(
        ali=ali,
        uploaded=uploaded,
        local_size=archive_size,
        require_remote_size=args.require_remote_size,
    )
    append_state(
        state_file,
        "upload_done",
        source=str(source),
        archive=str(archive_path),
        archive_bytes=archive_size,
        remote_file_id=object_get(uploaded, "file_id", "id"),
    )

    if args.keep_zips:
        logging.info("keeping local zip due to --keep-zips: %s", archive_path)
        return

    archive_path.unlink()
    append_state(state_file, "zip_removed", source=str(source), archive=str(archive_path))
    logging.info("removed local zip: %s", archive_path)


def main() -> int:
    args = parse_args()
    resume_state_file = Path(args.resume_state).expanduser() if args.resume_state else None
    batch_stamp = batch_stamp_from_state_file(resume_state_file) if resume_state_file else None
    batch_stamp = batch_stamp or now_stamp()
    staging_dir = Path(args.staging_dir).expanduser()
    staging_dir.mkdir(parents=True, exist_ok=True)

    log_suffix = f"{batch_stamp}.resume-{now_stamp()}" if resume_state_file else batch_stamp
    log_file = Path(args.log_file).expanduser() if args.log_file else staging_dir / f"alidrive_zip_upload.{log_suffix}.log"
    configure_logging(log_file)

    sources = [Path(item).expanduser() for item in (args.source or DEFAULT_SOURCES)]
    max_payload_bytes = gib_to_bytes(args.max_zip_gb)
    min_free_bytes = gib_to_bytes(args.min_free_gb)
    remote_dir = normalize_remote_path(
        args.remote_dir or f"/AutoDL-backups/{socket.gethostname()}-{batch_stamp}"
    )
    state_file = resume_state_file or staging_dir / f"alidrive_zip_upload.{batch_stamp}.jsonl"
    resume_state = load_resume_state(state_file) if resume_state_file else None

    logging.info("log file: %s", log_file)
    logging.info("state file: %s", state_file)
    if resume_state:
        logging.info(
            "resume state: uploaded=%d zip_done=%d",
            len(resume_state.uploaded_archives),
            len(resume_state.zip_done_archives),
        )
    logging.info("remote dir: %s", remote_dir)
    logging.info("staging dir: %s", staging_dir)
    logging.info("max zip payload: %s", format_gib(max_payload_bytes))
    logging.info("min free space: %s", format_gib(min_free_bytes))
    logging.info("sources: %s", ", ".join(str(source) for source in sources))

    validate_sources(sources, staging_dir)

    ali = None
    remote_parent_id = None
    if not args.dry_run:
        ali = import_aligo(args.aligo_config_dir, args.account_name)
        remote_parent_id = ensure_remote_folder(ali, remote_dir)
        logging.info("remote folder id: %s", remote_parent_id)
    else:
        logging.info("dry run enabled; no zip files will be created or uploaded")

    archive_counter = 0
    failures = 0

    for source in sources:
        logging.info("scanning source: %s", source)
        part_index = 0
        entries_iter = iter_source_entries(
            source=source,
            symlink_mode=args.symlink_mode,
            exclude_patterns=args.exclude,
        )
        for entries, payload_bytes in split_entries(entries_iter, max_payload_bytes):
            part_index += 1
            archive_counter += 1
            archive_path = staging_dir / archive_name(source, batch_stamp, part_index)

            logging.info(
                "planned archive: %s entries=%d payload=%s",
                archive_path.name,
                len(entries),
                format_gib(payload_bytes),
            )

            archive_key = str(archive_path)
            if resume_state and archive_key in resume_state.uploaded_archives:
                logging.info("resume skip uploaded archive: %s", archive_path)
                if args.stop_after and archive_counter >= args.stop_after:
                    logging.info("stop-after reached")
                    break
                continue

            if args.dry_run:
                if args.stop_after and archive_counter >= args.stop_after:
                    logging.info("stop-after reached")
                    return 0
                continue

            try:
                assert ali is not None
                assert remote_parent_id is not None
                if resume_state and archive_key in resume_state.zip_done_archives and archive_path.exists():
                    process_existing_archive(
                        args=args,
                        ali=ali,
                        source=source,
                        archive_path=archive_path,
                        remote_parent_id=remote_parent_id,
                        state_file=state_file,
                    )
                    resume_state = load_resume_state(state_file)
                else:
                    process_archive(
                        args=args,
                        ali=ali,
                        source=source,
                        entries=entries,
                        payload_bytes=payload_bytes,
                        archive_path=archive_path,
                        remote_parent_id=remote_parent_id,
                        state_file=state_file,
                        min_free_bytes=min_free_bytes,
                    )
                    if resume_state:
                        resume_state = load_resume_state(state_file)
            except Exception as exc:
                failures += 1
                append_state(
                    state_file,
                    "archive_failed",
                    source=str(source),
                    archive=str(archive_path),
                    error=repr(exc),
                )
                logging.exception("archive failed and local zip, if present, was kept: %s", archive_path)
                if not args.continue_on_error:
                    return 1

            if args.stop_after and archive_counter >= args.stop_after:
                logging.info("stop-after reached")
                break

        if args.stop_after and archive_counter >= args.stop_after:
            break

    logging.info("finished: archives=%d failures=%d", archive_counter, failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
