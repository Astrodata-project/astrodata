import os
from pathlib import Path

from git import GitCommandError

from astrodata.tracking._utils import get_tracked_files
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
        code_config = self.config.get("code", {})
        remote_name = code_config.get("remote", {}).get("name", "origin")
        branch = code_config.get("branch", "main")

        if self.code_tracker:
            self.code_tracker.align_with_remote()
            self.code_tracker.checkout(branch)
            if not self.code_tracker.pull(remote_name, branch):
                logger.error("Please resolve manually any conflicts before tracking.")
                return
        # self._pull_data()
        self._track_data()
        self._track_code()
        self._push_data()
        self._commit_and_push_code(commit_message)

    def _pull_data(self):
        """
        Pulls data files using the data tracker.
        """
        if not self.data_tracker:
            return
        logger.info("Pulling data from DVC remote...")
        self.data_tracker.pull()

    def _track_data(self):
        """
        Tracks data files using the data tracker.
        """
        if not self.data_tracker:
            return
        logger.info("Tracking data with DVC...")
        data_config = self.config.get("data", {})
        for path in data_config.get("paths", []):
            abs_path = (self.project_path / path).resolve()
            if abs_path.is_dir():
                for file in abs_path.rglob("*"):
                    if file.is_file() and file.suffix != ".dvc":
                        logger.info(f"Tracking file: {file}")
                        self.data_tracker.add(str(file.relative_to(self.project_path)))
            else:
                self.data_tracker.add(path)

    def _track_code(self):
        """
        Tracks code files using the code tracker, including handling remotes and branches.
        """
        if not self.code_tracker:
            return False
        logger.info("Tracking code with Git...")

        code_config = self.config.get("code", {})
        data_config = self.config.get("data", {})
        remote_name = code_config.get("remote", {}).get("name", "origin")
        if remote_name not in self.code_tracker.repo.remotes:
            remote_url = code_config.get("remote", {}).get("url")
            if remote_url:
                self.code_tracker.add_remote(remote_name, remote_url)
        final_files = get_tracked_files(
            self.project_path, code_config, self.data_tracker, data_config
        )
        self.code_tracker.add_to_index(final_files)
        self.code_tracker.remove_deleted_from_index()

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
