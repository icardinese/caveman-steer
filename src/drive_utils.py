"""Google Drive integration, chosen for zero setup friction on a fresh RunPod pod: gdown handles
restore with no OAuth screen at all (works off "anyone with the link" sharing), at the cost of backup
staying manual -- you zip results yourself and drag the file into Drive's web UI.

If this becomes annoying (e.g. you're doing this every session and want it automated end to end), the
documented upgrade path is a Google service account: create one in Google Cloud Console, download its
JSON key, share your Drive backup folder with the service account's email, then swap this module for
PyDrive2 or the raw Google API client using that key -- no interactive OAuth needed on any future pod
after that one-time setup. Not implemented here since it wasn't confirmed as wanted; this module is
intentionally the lower-friction path so nothing is blocked on that decision.
"""
import shutil
import zipfile
from pathlib import Path

from data_utils import RESULTS_DIR

try:
    import gdown
except ImportError:
    gdown = None


def restore_from_drive(file_id: str, dest_path: Path, force: bool = False) -> bool:
    """Downloads a single file from a Drive share link's file ID into dest_path, skipping the download
    entirely if dest_path already exists locally (mirrors the Stolfo pipeline's "skip if local,
    restore if backed up, else compute fresh" pattern -- here "compute fresh" just means whatever
    script would normally produce dest_path runs instead).

    file_id is the long string in a Drive share URL: https://drive.google.com/file/d/FILE_ID/view
    The file must be shared as "Anyone with the link" -- gdown has no auth flow and can't access
    anything more restricted than that.

    Returns True if a restore happened, False if skipped (already local) or gdown isn't installed.
    """
    if gdown is None:
        print("gdown not installed -- run `pip install gdown` first. Skipping restore, "
              "whatever step normally produces this file will need to run instead.")
        return False

    if dest_path.exists() and not force:
        print(f">>> {dest_path} already exists locally, skipping Drive restore")
        return False

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f">>> Restoring {dest_path.name} from Drive (file_id={file_id})")
    gdown.download(id=file_id, output=str(dest_path), quiet=False)
    return True


def zip_for_manual_backup(paths: list[Path], zip_name: str) -> Path:
    """Zips the given files/directories (relative to RESULTS_DIR, or absolute) into RESULTS_DIR/zip_name.
    Prints the resulting path with an explicit reminder that YOU still have to drag this into Drive --
    nothing here uploads it automatically. Pair with infra/backup_results.sh for a one-line call."""
    zip_path = RESULTS_DIR / zip_name
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            full_path = p if p.is_absolute() else RESULTS_DIR / p
            if not full_path.exists():
                print(f"WARNING: {full_path} does not exist, skipping")
                continue
            if full_path.is_dir():
                for f in full_path.rglob("*"):
                    if f.is_file():
                        zf.write(f, f.relative_to(RESULTS_DIR.parent))
            else:
                zf.write(full_path, full_path.relative_to(RESULTS_DIR.parent))

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f">>> Wrote {zip_path} ({size_mb:.1f} MB)")
    print(">>> MANUAL STEP: drag this file into your Drive backup folder -- nothing auto-uploads it.")
    return zip_path


if __name__ == "__main__":
    # Default: back up everything currently in results/ that isn't itself a previous backup zip.
    to_back_up = [p for p in RESULTS_DIR.iterdir() if not p.name.startswith("backup_") or not p.suffix == ".zip"]
    zip_for_manual_backup(to_back_up, "backup_results.zip")
