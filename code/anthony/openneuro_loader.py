from __future__ import annotations

from pathlib import Path
import argparse
import shutil
import subprocess


def run_command(command, cwd=None):
    command = [str(item) for item in command]
    print("$", " ".join(command))
    subprocess.run(
        command,
        cwd=str(cwd) if cwd is not None else None,
        check=True,
    )


def _require_program(program: str):
    if shutil.which(program) is None:
        raise RuntimeError(
            f"{program!r} is not installed. In Google Colab run:\n"
            "!apt-get update -qq\n"
            "!apt-get install -y -qq git-annex datalad"
        )


def clone_openneuro_dataset(
    dataset_id="ds003838",
    snapshot="1.0.6",
    local_root="/content",
):
    """
    Clone the OpenNeuro dataset repository and check out a fixed snapshot.
    """
    _require_program("datalad")
    _require_program("git")

    local_root = Path(local_root).expanduser()
    local_root.mkdir(parents=True, exist_ok=True)

    dataset_root = local_root / dataset_id
    repo_url = f"https://github.com/OpenNeuroDatasets/{dataset_id}.git"

    if not dataset_root.exists():
        run_command(
            ["datalad", "clone", repo_url, dataset_root]
        )
    else:
        print("Dataset repository already exists:", dataset_root)

    run_command(["git", "fetch", "--tags"], cwd=dataset_root)
    run_command(["git", "checkout", snapshot], cwd=dataset_root)

    return dataset_root


def find_eeg_subjects(dataset_root):
    dataset_root = Path(dataset_root)

    return sorted(
        folder.name.replace("sub-", "")
        for folder in dataset_root.glob("sub-*")
        if folder.is_dir() and (folder / "eeg").exists()
    )


def download_subject_eeg(dataset_root, subject):
    """
    Download one subject's EEG directory, including the .set/.fdt files and
    events.tsv needed by Workflow 3a.
    """
    dataset_root = Path(dataset_root)
    subject = str(subject).replace("sub-", "")
    eeg_relative = Path(f"sub-{subject}") / "eeg"

    run_command(
        ["datalad", "get", "-r", eeg_relative],
        cwd=dataset_root,
    )

    return dataset_root / eeg_relative


def download_openneuro_data(
    dataset_id="ds003838",
    snapshot="1.0.6",
    subjects="all",
    local_root="/content",
):
    """
    Download selected subjects from Kosachenko/OpenNeuro ds003838.
    Returns the dataset root directory for process_all_data().
    """
    dataset_root = clone_openneuro_dataset(
        dataset_id=dataset_id,
        snapshot=snapshot,
        local_root=local_root,
    )

    available_subjects = find_eeg_subjects(dataset_root)

    if subjects == "all":
        subjects_to_download = available_subjects
    else:
        subjects_to_download = [
            str(subject).replace("sub-", "")
            for subject in subjects
        ]

        unknown = sorted(
            set(subjects_to_download) - set(available_subjects)
        )
        if unknown:
            raise ValueError(
                "Unknown subject(s): " + ", ".join(unknown)
            )

    print("EEG subjects available:", len(available_subjects))
    print("Subjects selected:", len(subjects_to_download))

    for number, subject in enumerate(subjects_to_download, start=1):
        print(
            f"\nDownloading subject {number}/"
            f"{len(subjects_to_download)}: sub-{subject}"
        )
        download_subject_eeg(dataset_root, subject)

    set_files = sorted(dataset_root.rglob("*_eeg.set"))
    event_files = sorted(dataset_root.rglob("*_events.tsv"))

    print("\nDownloaded EEG .set files:", len(set_files))
    print("Event TSV files available:", len(event_files))

    return dataset_root


def _main():
    parser = argparse.ArgumentParser(
        description="Download Kosachenko/OpenNeuro EEG data."
    )
    parser.add_argument("--dataset-id", default="ds003838")
    parser.add_argument("--snapshot", default="1.0.6")
    parser.add_argument("--local-root", default="/content")
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=["032"],
        help="Subject IDs, or use --all.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download every available EEG subject.",
    )

    args = parser.parse_args()
    subjects = "all" if args.all else args.subjects

    download_openneuro_data(
        dataset_id=args.dataset_id,
        snapshot=args.snapshot,
        subjects=subjects,
        local_root=args.local_root,
    )


if __name__ == "__main__":
    _main()
