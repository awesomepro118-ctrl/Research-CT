from pathlib import Path
import subprocess


def run_command(command, cwd=None):
    command = [str(item) for item in command]
    print("$", " ".join(command))

    subprocess.run(
        command,
        cwd=str(cwd) if cwd is not None else None,
        check=True,
    )


def clone_openneuro_dataset(
    dataset_id="ds003838",
    snapshot="1.0.6",
    local_root="/content",
):
    local_root = Path(local_root)
    dataset_root = local_root / dataset_id
    repo_url = f"https://github.com/OpenNeuroDatasets/{dataset_id}.git"

    if not dataset_root.exists():
        run_command(
            [
                "datalad",
                "clone",
                repo_url,
                dataset_root,
            ]
        )
    else:
        print("Dataset repository already exists:", dataset_root)

    run_command(
        ["git", "fetch", "--tags"],
        cwd=dataset_root,
    )

    run_command(
        ["git", "checkout", snapshot],
        cwd=dataset_root,
    )

    return dataset_root


def find_eeg_subjects(dataset_root):
    dataset_root = Path(dataset_root)

    subjects = sorted(
        folder.name.replace("sub-", "")
        for folder in dataset_root.glob("sub-*")
        if folder.is_dir()
        and (folder / "eeg").exists()
    )

    return subjects


def download_subject_eeg(
    dataset_root,
    subject,
):
    dataset_root = Path(dataset_root)

    subject = str(subject).replace("sub-", "")
    eeg_relative = Path(f"sub-{subject}") / "eeg"

    run_command(
        [
            "datalad",
            "get",
            "-r",
            eeg_relative,
        ],
        cwd=dataset_root,
    )

    return dataset_root / eeg_relative


def download_openneuro_data(
    dataset_id="ds003838",
    snapshot="1.0.6",
    subjects="all",
    local_root="/content",
):
    dataset_root = clone_openneuro_dataset(
        dataset_id=dataset_id,
        snapshot=snapshot,
        local_root=local_root,
    )

    available_subjects = find_eeg_subjects(
        dataset_root
    )

    if subjects == "all":
        subjects_to_download = available_subjects
    else:
        subjects_to_download = [
            str(subject).replace("sub-", "")
            for subject in subjects
        ]

    print("EEG subjects available:", len(available_subjects))
    print("Subjects selected:", len(subjects_to_download))

    for number, subject in enumerate(
        subjects_to_download,
        start=1,
    ):
        print()
        print(
            "Downloading subject",
            number,
            "/",
            len(subjects_to_download),
            ": sub-" + subject,
        )

        download_subject_eeg(
            dataset_root,
            subject,
        )

    set_files = sorted(
        dataset_root.rglob("*.set")
    )

    event_files = sorted(
        dataset_root.rglob("*events.tsv")
    )

    print()
    print("Downloaded .set files:", len(set_files))
    print("Event TSV files available:", len(event_files))

    return dataset_root
