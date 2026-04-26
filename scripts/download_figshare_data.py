#!/usr/bin/env python3
"""Reviewer-friendly Figshare data downloader.

Fetches the hawk kinematic morphospace dataset (~2.3 GB) into ./data/.

Two modes:
    Embargoed / private : during peer review, paste the share token from
                          the manuscript's Data Availability section.
                          The full share URL works too (the token is the
                          string after ``https://figshare.com/s/``).
    Public              : once the dataset is published, no token needed.

Token resolution order (first hit wins):
    1. ``--token`` CLI flag
    2. ``FIGSHARE_PRIVATE_TOKEN`` environment variable
    3. ``scripts/.figshare_token`` (gitignored)
    4. interactive prompt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import requests
from tqdm import tqdm

ARTICLE_ID = 32101528
NDOWNLOADER = "https://ndownloader.figshare.com/files/{file_id}"
ARTICLE_API = "https://api.figshare.com/v2/articles/{article_id}"

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
SCRIPTS_DIR = REPO_ROOT / "scripts"
TOKEN_FILE = SCRIPTS_DIR / ".figshare_token"
MANIFEST_FILE = SCRIPTS_DIR / "_figshare_manifest.json"
CHUNK = 1 << 20  # 1 MiB

TOKEN_RE = re.compile(r"(?:figshare\.com/s/)?([a-fA-F0-9]{20,})")


def banner() -> None:
    """Print the welcome banner."""
    print(
        "\n"
        "  ╭─────────────────────────────────────────────────────────╮\n"
        "  │  Hawk kinematic morphospace — dataset downloader        │\n"
        "  │                                                         │\n"
        "  │  Fetches ~2.3 GB into ./data/                           │\n"
        "  ╰─────────────────────────────────────────────────────────╯\n"
    )


def parse_token(raw: str | None) -> str | None:
    """Pull a Figshare private-link token out of a string.

    Accepts the bare hex token or a full share URL.
    """
    if not raw:
        return None
    m = TOKEN_RE.search(raw.strip())
    return m.group(1).lower() if m else None


def resolve_token(cli_token: str | None, no_prompt: bool) -> str | None:
    """Find a token via CLI flag, env var, file, or prompt."""
    if cli_token:
        tok = parse_token(cli_token)
        if tok:
            return tok
        print("  ✗ --token did not look like a valid Figshare token.")

    env = os.environ.get("FIGSHARE_PRIVATE_TOKEN")
    if env:
        tok = parse_token(env)
        if tok:
            print("  ✓ using token from FIGSHARE_PRIVATE_TOKEN env var")
            return tok
        print("  ✗ FIGSHARE_PRIVATE_TOKEN set but unparseable; ignoring.")

    if TOKEN_FILE.is_file():
        tok = parse_token(TOKEN_FILE.read_text())
        if tok:
            print(f"  ✓ using token from {TOKEN_FILE.relative_to(REPO_ROOT)}")
            return tok
        print(f"  ✗ {TOKEN_FILE} present but unparseable; ignoring.")

    if no_prompt:
        return None
    return prompt_for_token()


def prompt_for_token() -> str | None:
    """Interactive token prompt with retry-on-bad-input loop."""
    print(
        "\n  Where to find the token:\n"
        "    Open the manuscript PDF → Data Availability section →\n"
        "    copy the URL after 'https://figshare.com/s/'.\n"
        "    Either the bare token or the full URL works.\n"
        "    (Press Enter with nothing to try the public endpoint instead.)\n"
    )
    for _ in range(5):
        try:
            raw = input("  Paste token or URL: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if not raw:
            return None
        tok = parse_token(raw)
        if tok:
            return tok
        print("  ✗ that does not look like a Figshare token (expected hex string ≥20 chars). Try again.")
    print("  ✗ too many attempts; aborting.")
    sys.exit(1)


def offer_to_persist(token: str) -> None:
    """Ask whether to save the token locally for future runs."""
    if TOKEN_FILE.is_file():
        return
    try:
        ans = input(
            f"\n  Save token to {TOKEN_FILE.relative_to(REPO_ROOT)} so you "
            "won't be asked again? [Y/n]: "
        ).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return
    if ans in {"", "y", "yes"}:
        TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        TOKEN_FILE.write_text(token + "\n")
        TOKEN_FILE.chmod(0o600)
        print(f"  ✓ saved to {TOKEN_FILE.relative_to(REPO_ROOT)} (gitignored)")


def load_manifest() -> dict[str, dict]:
    """Read the bundled file manifest (id/size/md5 per file)."""
    if not MANIFEST_FILE.is_file():
        print(f"  ✗ manifest missing: {MANIFEST_FILE}")
        sys.exit(1)
    return json.loads(MANIFEST_FILE.read_text())


def fetch_public_manifest() -> dict[str, dict] | None:
    """Pull the file list from the public article API (post-embargo)."""
    try:
        r = requests.get(ARTICLE_API.format(article_id=ARTICLE_ID), timeout=30)
    except requests.RequestException as exc:
        print(f"  ✗ network error contacting Figshare: {exc}")
        return None
    if not r.ok:
        return None
    files = r.json().get("files", [])
    if not files:
        return None
    return {
        f["name"]: {
            "id": int(f["id"]),
            "size": int(f["size"]),
            "md5": f.get("supplied_md5") or f.get("computed_md5") or "",
        }
        for f in files
    }


def md5_of(path: Path) -> str:
    """Return the MD5 of an existing local file."""
    h = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def needs_download(dst: Path, expected_md5: str, expected_size: int) -> bool:
    """Decide whether to (re-)download a file."""
    if not dst.is_file():
        return True
    if dst.stat().st_size != expected_size:
        return True
    if expected_md5 and md5_of(dst) != expected_md5:
        return True
    return False


def download_one(name: str, meta: dict, token: str | None) -> bool:
    """Download a single file. Returns True on success, False on auth fail."""
    dst = DATA_DIR / name
    size = meta["size"]
    expected_md5 = meta.get("md5", "")

    if not needs_download(dst, expected_md5, size):
        print(f"  ✓ {name} ({size / 1e6:.1f} MB) — already present, skipping")
        return True

    dst.parent.mkdir(parents=True, exist_ok=True)
    url = NDOWNLOADER.format(file_id=meta["id"])
    params = {"private_link": token} if token else {}

    try:
        with requests.get(url, params=params, stream=True, timeout=60) as r:
            if r.status_code in (401, 403):
                print(f"  ✗ {name}: auth failed (HTTP {r.status_code}). Token wrong?")
                return False
            r.raise_for_status()
            tmp = dst.with_suffix(dst.suffix + ".part")
            with tmp.open("wb") as fh, tqdm(
                total=size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=f"  {name}",
                leave=False,
            ) as bar:
                for block in r.iter_content(CHUNK):
                    fh.write(block)
                    bar.update(len(block))
            tmp.replace(dst)
    except requests.RequestException as exc:
        print(f"  ✗ {name}: download error: {exc}")
        return False

    if expected_md5:
        actual = md5_of(dst)
        if actual != expected_md5:
            print(f"  ✗ {name}: MD5 mismatch (got {actual}, expected {expected_md5})")
            dst.unlink(missing_ok=True)
            return False
    print(f"  ✓ {name} ({size / 1e6:.1f} MB)")
    return True


def download_all(manifest: dict[str, dict], token: str | None) -> int:
    """Download every file in the manifest. Returns count of failures."""
    total_bytes = sum(m["size"] for m in manifest.values())
    print(f"\n  {len(manifest)} files, {total_bytes / 1e9:.2f} GB total\n")
    failed = 0
    for i, (name, meta) in enumerate(sorted(manifest.items()), 1):
        print(f"  [{i}/{len(manifest)}]")
        if not download_one(name, meta, token):
            failed += 1
    return failed


def probe_auth(token: str | None, manifest: dict[str, dict]) -> bool:
    """HEAD a small file first to fail fast on bad token."""
    name, meta = min(manifest.items(), key=lambda kv: kv[1]["size"])
    url = NDOWNLOADER.format(file_id=meta["id"])
    params = {"private_link": token} if token else {}
    try:
        r = requests.head(url, params=params, timeout=15, allow_redirects=False)
    except requests.RequestException as exc:
        print(f"  ✗ network error: {exc}")
        return False
    if r.status_code in (200, 302, 303):
        return True
    if r.status_code in (401, 403):
        print(f"  ✗ Figshare rejected the token (HTTP {r.status_code}).")
        return False
    print(f"  ✗ unexpected response (HTTP {r.status_code}) probing {name}.")
    return False


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--token",
        help="Figshare private-link token (or full share URL). "
        "Overrides env var, file, and prompt.",
    )
    p.add_argument(
        "--no-prompt",
        action="store_true",
        help="Do not prompt for a token; fail if none found in flag/env/file.",
    )
    p.add_argument(
        "--refresh-manifest",
        action="store_true",
        help="Pull the file list from the public article API instead of the bundled manifest.",
    )
    args = p.parse_args()

    banner()

    manifest: dict[str, dict] | None = None
    if args.refresh_manifest:
        print("  Fetching live manifest from public article API…")
        manifest = fetch_public_manifest()
        if manifest is None:
            print("  ✗ public manifest unavailable; falling back to bundled manifest.")
    if manifest is None:
        manifest = load_manifest()
        print(f"  ✓ loaded bundled manifest ({len(manifest)} files)")

    token = resolve_token(args.token, args.no_prompt)
    if token is None:
        print("\n  No token supplied — trying public download (works only post-embargo).\n")
    else:
        print(f"\n  ✓ token resolved (…{token[-6:]})\n")

    print("  Probing Figshare with smallest file…")
    if not probe_auth(token, manifest):
        if token:
            print(
                "\n  Token check failed. Re-run and paste the token again,\n"
                "  or pass --token <value>.\n"
            )
        else:
            print(
                "\n  Public access not yet available. Re-run with the private\n"
                "  share token from the manuscript's Data Availability section.\n"
            )
        sys.exit(1)
    print("  ✓ access confirmed\n")

    failed = download_all(manifest, token)

    if failed:
        print(f"\n  ✗ {failed} file(s) failed. Re-run to resume — completed files will be skipped.")
        sys.exit(1)

    if token and not args.no_prompt:
        offer_to_persist(token)

    print(
        "\n  ✓ All files downloaded to ./data/\n"
        "  Next: uv run jupyter lab → open notebooks/NB00 first.\n"
    )


if __name__ == "__main__":
    main()
