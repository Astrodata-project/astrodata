import os
from pathlib import Path

from astrodata.tracking.utils import get_tracked_files
from astrodata.utils.logger import setup_logger
from astrodata.utils.utils import read_config

from .CodeTracking import CodeTracker
from .DataTracking import DataTracker

logger = setup_logger(__name__)


class Tracker:
    def __init__(self, config_path: str):
        self.config = read_config(config_path)
        self.project_path = Path(self.config["project_path"]).resolve()

        self.code_tracker = None
        self.data_tracker = None

        if self.config.get("code", {}).get("enable", False):
            ssh_key = self.config.get("code", {}).get("ssh_key_path")
            token = self.config.get("code", {}).get("token")
            branch = self.config.get("code", {}).get("branch", "main")
            self.code_tracker = CodeTracker(
                self.project_path, ssh_key_path=ssh_key, token=token, branch=branch
            )
        if self.config.get("data", {}).get("enable", False):
            remote = self.config.get("data", {}).get("remote", "myremote")
            self.data_tracker = DataTracker(self.project_path, remote)

    def track(self, commit_message: str = None):
        """
        Orchestrates the tracking of data and code, pushing data and committing code.
        """
        if self.code_tracker:
            self.code_tracker.align_with_remote()
        self._track_data()
        self._track_code()
        self._push_data()
        self._commit_and_push_code(commit_message)

    def _track_data(self):
        """
        Tracks data files using the data tracker.
        """
        if not self.data_tracker:
            return
        logger.info("Tracking data with DVC...")
        data_config = self.config.get("data", {})
        for path in data_config.get("paths", []):
            self.data_tracker.add(path)

    def _track_code(self):
        """
        Tracks code files using the code tracker, including handling remotes and branches.
        """
        if not self.code_tracker:
            return

        code_config = self.config.get("code", {})
        data_config = self.config.get("data", {})
        remote_name = code_config.get("remote", {}).get("name", "origin")
        branch = code_config.get("branch", "main")
        if remote_name not in self.code_tracker.repo.remotes:
            remote_url = code_config.get("remote", {}).get("url")
            if remote_url:
                self.code_tracker.add_remote(remote_name, remote_url)
        self.code_tracker.checkout(branch)
        self.code_tracker.pull(remote_name, branch)

        logger.info("Tracking code with Git...")
        final_files = get_tracked_files(
            self.project_path, code_config, self.data_tracker, data_config
        )
        self.code_tracker.add_to_index(final_files)

    def _get_tracked_files(self, code_config):
        """
        Returns a list of files to be tracked by git, considering tracked_files and .gitignore.
        """
        all_files = [
            os.path.join(dirpath, f)
            for (dirpath, dirnames, filenames) in os.walk(self.project_path)
            for f in filenames
        ]
        tracked_files = code_config.get("tracked_files", ["src", "pyproject.toml"])
        if self.data_tracker:
            tracked_files.extend(["/.dvc", "/.dvcignore", "/.gitignore"])
            for path in self.config.get("data", {}).get("paths", []):
                tracked_files.append(path + ".dvc")
        tracked_files = [t if t.startswith("/") else "/" + t for t in tracked_files]
        gitignore_files = []
        for f in all_files:
            if ".gitignore" in f:
                with open(f, "r") as file:
                    ignored = file.read().splitlines()
                gitignore_files.extend([f.split(".gitignore")[0] + i for i in ignored])
        final_files = all_files[:]
        for f in final_files[:]:
            if not any(t in f for t in tracked_files):
                final_files.remove(f)
            if any(g in f for g in gitignore_files):
                final_files.remove(f)
        return final_files

    def _push_data(self):
        """
        Pushes tracked data to the DVC remote.
        """
        if not self.data_tracker:
            return
        logger.info("Pushing data to DVC remote...")
        self.data_tracker.push()

    def _commit_and_push_code(self, commit_message: str = None):
        """
        Commits and pushes code changes to the git remote.
        """
        if not self.code_tracker:
            return
        logger.info("Committing and pushing code to Git remote...")
        code_config = self.config.get("code", {})
        if not commit_message:
            commit_message = code_config.get(
                "commit_message", "Auto commit by astrodata"
            )
        remote = code_config.get("remote", {})
        remote_name = remote.get("name", "origin")
        remote_url = remote.get("url")
        if self.code_tracker.create_commit(commit_message):
            self.code_tracker.push(remote_name, self.code_tracker.branch, remote_url)
