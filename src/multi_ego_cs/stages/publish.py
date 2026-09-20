"""Stage 08 - upload a release tree to the Hugging Face Hub.

Uploading a 100k-file, several-hundred-GB dataset over one commit is a good way
to lose six hours to a dropped socket, so this stage:

* uploads in **bounded commits** (``publish.files_per_commit``), so an
  interruption costs one chunk rather than the whole run;
* **skips files already on the Hub** by comparing paths against the remote
  listing, which makes a re-run a cheap resume;
* orders uploads **small-to-large** (manifests, metadata, parquet, then video),
  so the dataset is browsable and partially usable early;
* refuses to run without an explicit ``--yes``, because publishing is public,
  attributable, and not something to trigger by accident.

Requires a write-scoped token in ``HF_TOKEN``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..config import Config
from ..util.io_utils import du_bytes, human_bytes
from ..util.logging_setup import get_logger

log = get_logger("publish")

# Uploaded in this order: cheap and structural first, bulk video last.
_DEFAULT_ORDER = ["README.md", "manifest", "metadata", "align", "state_action", "video", "demo"]


def _iter_local_files(release: Path, include: list[str]) -> list[tuple[Path, str]]:
    """(absolute path, repo-relative path) for everything selected by `include`."""
    out: list[tuple[Path, str]] = []
    seen: set[str] = set()
    ordered = [i for i in _DEFAULT_ORDER if i in include] + [
        i for i in include if i not in _DEFAULT_ORDER
    ]
    for item in ordered:
        target = release / item
        if target.is_file():
            rel = item
            if rel not in seen:
                seen.add(rel)
                out.append((target, rel))
        elif target.is_dir():
            for path in sorted(target.rglob("*")):
                if not path.is_file() or path.name.startswith("."):
                    continue
                rel = path.relative_to(release).as_posix()
                if rel not in seen:
                    seen.add(rel)
                    out.append((path, rel))
        else:
            log.debug("include entry %r not present in %s", item, release)
    return out


def _remote_files(api: Any, repo_id: str, repo_type: str, revision: str) -> set[str]:
    try:
        return set(api.list_repo_files(repo_id=repo_id, repo_type=repo_type, revision=revision))
    except Exception as exc:
        log.warning("Could not list remote files (%s); treating repo as empty.", exc)
        return set()


def run(
    cfg: Config,
    yes: bool = False,
    dry_run: bool = False,
    include: list[str] | None = None,
    revision: str | None = None,
) -> dict[str, Any]:
    from huggingface_hub import CommitOperationAdd, HfApi

    pcfg = cfg.publish
    layout = cfg.layout
    release = layout.release

    if not pcfg.repo_id:
        raise ValueError("publish.repo_id is unset (or export MECS_REPO_ID=owner/name)")
    if not release.exists():
        raise FileNotFoundError(f"No release tree at {release} - run `mecs package` first.")

    token = cfg.hf_token
    if not token:
        raise RuntimeError(
            "No Hugging Face token. Export HF_TOKEN with a *write*-scoped token "
            "(https://huggingface.co/settings/tokens)."
        )

    if pcfg.enable_hf_transfer:
        try:
            import hf_transfer  # noqa: F401

            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
            log.info("hf_transfer enabled")
        except ImportError:
            log.info("hf_transfer not installed; using the default uploader "
                     "(`pip install 'multi-ego-cs[fast-upload]'` to speed this up)")

    rev = revision or pcfg.revision
    selected = include or pcfg.include
    files = _iter_local_files(release, selected)
    if not files:
        log.error("Nothing to upload from %s with include=%s", release, selected)
        return {"uploaded": 0}

    total_bytes = sum(p.stat().st_size for p, _ in files)
    log.info(
        "Local release: %d files, %s (%s)",
        len(files), human_bytes(total_bytes), human_bytes(du_bytes(release)),
    )

    api = HfApi(token=token)
    who = api.whoami()
    log.info("Authenticated as %s", who.get("name"))

    try:
        api.repo_info(repo_id=pcfg.repo_id, repo_type=pcfg.repo_type)
        exists = True
    except Exception:
        exists = False

    if not exists:
        log.info("Repo %s does not exist; it will be created (private=%s)",
                 pcfg.repo_id, pcfg.private)

    remote = _remote_files(api, pcfg.repo_id, pcfg.repo_type, rev) if exists else set()
    pending = [(p, rel) for p, rel in files if rel not in remote]
    pending_bytes = sum(p.stat().st_size for p, _ in pending)

    log.info(
        "Remote has %d files; %d local files already present, %d to upload (%s)",
        len(remote), len(files) - len(pending), len(pending), human_bytes(pending_bytes),
    )

    if not pending:
        log.info("Remote already has every local file - nothing to do.")
        return {"uploaded": 0, "skipped": len(files)}

    by_prefix: dict[str, int] = {}
    for _, rel in pending:
        by_prefix[rel.split("/")[0]] = by_prefix.get(rel.split("/")[0], 0) + 1
    log.info("Pending by top-level path: %s", by_prefix)

    if dry_run:
        log.info("[dry-run] would upload %d files (%s) to %s@%s in %d commit(s)",
                 len(pending), human_bytes(pending_bytes), pcfg.repo_id, rev,
                 -(-len(pending) // pcfg.files_per_commit))
        for _, rel in pending[:15]:
            log.info("  [dry-run] %s", rel)
        if len(pending) > 15:
            log.info("  [dry-run] … and %d more", len(pending) - 15)
        return {"pending": len(pending), "bytes": pending_bytes, "dry_run": True}

    if not yes:
        raise RuntimeError(
            f"Refusing to publish without confirmation.\n"
            f"  repo:   {pcfg.repo_id} ({pcfg.repo_type}, "
            f"{'private' if pcfg.private else 'PUBLIC'})\n"
            f"  files:  {len(pending)}  ({human_bytes(pending_bytes)})\n"
            f"Re-run with --yes to proceed, or --dry-run to inspect the plan."
        )

    if not exists:
        api.create_repo(
            repo_id=pcfg.repo_id, repo_type=pcfg.repo_type, private=pcfg.private, exist_ok=True
        )

    chunk_size = max(1, pcfg.files_per_commit)
    chunks = [pending[i : i + chunk_size] for i in range(0, len(pending), chunk_size)]
    log.info("Uploading in %d commit(s) of up to %d files", len(chunks), chunk_size)

    uploaded = 0
    for index, chunk in enumerate(chunks, start=1):
        operations = [
            CommitOperationAdd(path_in_repo=rel, path_or_fileobj=str(path))
            for path, rel in chunk
        ]
        chunk_bytes = sum(p.stat().st_size for p, _ in chunk)
        log.info(
            "commit %d/%d: %d files, %s",
            index, len(chunks), len(chunk), human_bytes(chunk_bytes),
        )
        api.create_commit(
            repo_id=pcfg.repo_id,
            repo_type=pcfg.repo_type,
            revision=rev,
            operations=operations,
            commit_message=f"{pcfg.commit_message} ({index}/{len(chunks)})",
        )
        uploaded += len(chunk)
        log.info("  done - %d/%d files uploaded", uploaded, len(pending))

    log.info("Published %d files to https://huggingface.co/datasets/%s", uploaded, pcfg.repo_id)
    return {"uploaded": uploaded, "commits": len(chunks), "bytes": pending_bytes}


def upload_card(cfg: Config, card_path: str | Path, yes: bool = False) -> dict[str, Any]:
    """Upload only the dataset card (README.md). Cheap and separately reviewable."""
    from huggingface_hub import HfApi

    pcfg = cfg.publish
    path = Path(card_path)
    if not path.exists():
        raise FileNotFoundError(path)
    token = cfg.hf_token
    if not token:
        raise RuntimeError("No Hugging Face token; export HF_TOKEN.")
    if not yes:
        raise RuntimeError(
            f"Refusing to overwrite the dataset card on {pcfg.repo_id} without --yes."
        )

    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo="README.md",
        repo_id=pcfg.repo_id,
        repo_type=pcfg.repo_type,
        commit_message="Update dataset card",
    )
    log.info("Dataset card updated on %s", pcfg.repo_id)
    return {"ok": True}
