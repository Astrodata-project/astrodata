import os
from pathlib import Path

from git import GitCommandError

from astrodata.tracking._utils import get_tracked_files
from astrodata.utils.logger import setup_logger
from astrodata.utils.utils import read_config

from .CodeTracker import CodeTracker
from .DataTracker import DataTracker

logger = setup_logger(__name__)


class Tracker:
    """
    Orchestrates code and data tracking for a project using Git and DVC.

    This class manages both code and data versioning, providing methods to track,
    commit, and push changes to remote repositories for reproducible research.

    Args:
        config_path (str): Path to the configuration file.
    """

    def __init__(self, config_path: str):
        """
        Initialize the Tracker with the given configuration file.

        Args:
            config_path (str): Path to the configuration file.
        """
        self.config = read_config(config_path)
        self.project_path = self.config["project_path"]

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
        Orchestrate the tracking of data and code, pushing data and committing code.

        This method aligns the code repository with the remote, tracks data and code changes,
        pushes data to the DVC remote, and commits and pushes code changes to the Git remote.

        Args:
            commit_message (str, optional): Commit message for the code changes.
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
        Pull data files using the data tracker.

        This method pulls tracked data from the configured DVC remote storage.
        """
        if not self.data_tracker:
            return
        logger.info("Pulling data from DVC remote...")
        self.data_tracker.pull()

    def _track_data(self):
        """
        Track data files using the data tracker.

        This method adds specified data files or directories to DVC tracking,
        based on the configuration.
        """
        if not self.data_tracker:
            return
        logger.info("Tracking data with DVC...")
        data_config = self.config.get("data", {})
        paths = data_config.get("paths", [])
        paths.append("astrodata_files")
        for path in data_config.get("paths", []):
            abs_path = (self.project_path / path).resolve()
            if abs_path.is_dir():
                for file in abs_path.rglob("*"):
                    if (
                        file.is_file()
                        and file.suffix != ".dvc"
                        and file.name != ".gitignore"
                    ):
                        logger.info(f"Tracking file: {file}")
                        self.data_tracker.add(str(file.relative_to(self.project_path)))
            else:
                self.data_tracker.add(path)

    def _track_code(self):
        """
        Track code files using the code tracker, including handling remotes and branches.

        This method ensures the code repository is properly configured with remotes,
        adds tracked files to the Git index, and removes deleted files from the index.
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
        Push tracked data to the DVC remote.

        This method uploads tracked data files to the configured DVC remote storage.
        """
        if not self.data_tracker:
            return
        logger.info("Pushing data to DVC remote...")
        self.data_tracker.push()

    def _commit_and_push_code(self, commit_message: str = None):
        """
        Commit and push code changes to the Git remote.

        This method creates a commit with the specified message and pushes it to the
        configured Git remote and branch.

        Args:
            commit_message (str, optional): Commit message for the code changes.
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
