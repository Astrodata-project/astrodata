import os
from pathlib import Path


def get_tracked_files(project_path, code_config, data_tracker=None, data_config=None):
    """
    Returns a list of files to be tracked by git, considering tracked_files and .gitignore.
    """
    all_files = [
        os.path.join(dirpath, f)
        for (dirpath, dirnames, filenames) in os.walk(project_path)
        for f in filenames
    ]
    tracked_files = code_config.get("tracked_files", ["src", "pyproject.toml"])
    if data_tracker and data_config:
        tracked_files.extend(["/.dvc", "/.dvcignore", "/.gitignore"])
    tracked_files = [t if t.startswith("/") else "/" + t for t in tracked_files]
    gitignore_files = []
    for f in all_files:
        if ".gitignore" in f:
            with open(f, "r") as file:
                ignored = file.read().splitlines()
            gitignore_files.extend([f.split(".gitignore")[0] + i for i in ignored])
    gitignore_files = [g.replace("//", "/") for g in gitignore_files]
    final_files = all_files[:]
    for f in final_files[:]:
        if not any(t in f for t in tracked_files):
            final_files.remove(f)
            continue
        if any(g in f for g in gitignore_files):
            final_files.remove(f)
    if data_tracker and data_config:
        for path in data_config.get("paths", []):
            abs_path = (Path(project_path) / path).resolve()
            if abs_path.is_dir():
                for file in abs_path.rglob("*.dvc"):
                    final_files.append(str(file))
            else:
                final_files.append(str(project_path) + "/" + path + ".dvc")
    final_files = [f.replace("//", "/") for f in final_files]
    return final_files
