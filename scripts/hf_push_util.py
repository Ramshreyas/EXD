#!/usr/bin/env python3
"""
HF Push Utility — Quick uploads from the local workbench to the EXDai org.

Usage:
    python3 scripts/hf_push_util.py <repo_id> <local_path> [--to <hub_path>]
    python3 scripts/hf_push_util.py --create <repo_id> --type <model|dataset|space> [--sdk <sdk>]

Examples:
    # Upload a single file
    python3 scripts/hf_push_util.py EXDai/benchmark-results ./bench.txt --to runs/bench.txt

    # Create a new Space
    python3 scripts/hf_push_util.py --create EXDai/my-space --type space --sdk static

    # Upload an entire directory
    python3 scripts/hf_push_util.py EXDai/my-dataset ./data/ --recursive
"""

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, login
from huggingface_hub.errors import BadRequestError, RepositoryNotFoundError


def get_api() -> HfApi:
    """Authenticate and return an HfApi instance."""
    api = HfApi()
    try:
        user = api.whoami()
        print(f"✓ Authenticated as @{user['name']}", file=sys.stderr)
    except Exception:
        print("✗ Not authenticated. Set HF_TOKEN or run `hf auth login`.", file=sys.stderr)
        sys.exit(1)
    return api


def resolve_repo_type(repo_id: str) -> str:
    """Infer repo type from repo_id or default to 'model'."""
    if repo_id.startswith("datasets/"):
        return "dataset"
    if repo_id.startswith("spaces/"):
        return "space"
    return "model"


def cmd_create(args):
    """Create a new repo."""
    api = get_api()
    repo_type = args.type or resolve_repo_type(args.repo_id)

    kwargs = dict(
        repo_id=args.repo_id,
        repo_type=repo_type,
        exist_ok=args.exist_ok,
        private=args.private,
    )
    if repo_type == "space" and args.sdk:
        kwargs["space_sdk"] = args.sdk

    try:
        repo = api.create_repo(**kwargs)
        url = (
            f"https://huggingface.co/{'spaces/' if repo_type == 'space' else 'datasets/' if repo_type == 'dataset' else ''}{args.repo_id}"
        )
        print(f"✓ Created {repo_type}: {url}")
        return repo
    except BadRequestError as e:
        print(f"✗ Failed: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_upload(args):
    """Upload file(s) to a repo."""
    api = get_api()
    repo_type = args.type or resolve_repo_type(args.repo_id)

    local_path = Path(args.local_path)
    if not local_path.exists():
        print(f"✗ Local path not found: {local_path}", file=sys.stderr)
        sys.exit(1)

    if args.recursive and local_path.is_dir():
        uploaded = 0
        for fpath in local_path.rglob("*"):
            if fpath.is_file():
                rel = fpath.relative_to(local_path)
                hub_path = str(rel)
                if args.to:
                    hub_path = f"{args.to.rstrip('/')}/{hub_path}"
                api.upload_file(
                    path_or_fileobj=str(fpath),
                    path_in_repo=hub_path,
                    repo_id=args.repo_id,
                    repo_type=repo_type,
                )
                uploaded += 1
                print(f"  ✓ {hub_path}")
        print(f"✓ Uploaded {uploaded} files to {args.repo_id}")
    else:
        hub_path = args.to or local_path.name
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=hub_path,
            repo_id=args.repo_id,
            repo_type=repo_type,
        )
        print(f"✓ Uploaded {local_path.name} → {args.repo_id}/{hub_path}")


def main():
    parser = argparse.ArgumentParser(
        description="HF Push Utility — upload assets to the EXDai org.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Upload subcommand
    up = sub.add_parser("upload", help="Upload file(s) to a repo")
    up.add_argument("repo_id", help="Hub repo ID (e.g. EXDai/my-repo)")
    up.add_argument("local_path", help="Local file or directory path")
    up.add_argument("--to", help="Path in repo (default: basename of local_path)")
    up.add_argument("--type", choices=["model", "dataset", "space"], help="Repo type (auto-detected from repo_id if omitted)")
    up.add_argument("--recursive", "-r", action="store_true", help="Upload directory recursively")
    up.set_defaults(func=cmd_upload)

    # Create subcommand
    cr = sub.add_parser("create", help="Create a new repo")
    cr.add_argument("repo_id", help="Hub repo ID (e.g. EXDai/my-repo)")
    cr.add_argument("--type", choices=["model", "dataset", "space"], default="model", help="Repo type")
    cr.add_argument("--sdk", help="Space SDK (required for spaces: static, gradio, streamlit)")
    cr.add_argument("--private", action="store_true", help="Create private repo")
    cr.add_argument("--exist-ok", action="store_true", help="Don't error if repo exists")
    cr.set_defaults(func=cmd_create)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
