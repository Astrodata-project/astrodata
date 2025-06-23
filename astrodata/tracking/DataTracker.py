import logging
from pathlib import Path

from dvc.config import Config
from dvc.output import OutputAlreadyTrackedError
from dvc.repo import Repo as DvcRepo

logger = logging.getLogger(__name__)


class DataTracker:
    """
    A class to manage and track data files using DVC (Data Version Control).

    This class provides methods to initialize or open a DVC repository, configure remotes,
    add files to DVC tracking, and synchronize data with a remote storage.

    Args:
        repo_path (str): Path to the local DVC repository.
        remote (str): URL of the DVC remote storage.
    """

    def __init__(self, repo_path: str, remote: str):
        """
        Initialize the DataTracker with a DVC repository and configure the remote.

        Args:
            repo_path (str): Path to the local DVC repository.
            remote (str): URL of the DVC remote storage.
        """
        self.repo_path = Path(repo_path).resolve()
        try:
            self.dvc_repo = DvcRepo(self.repo_path)
            logger.info(f"Using existing DVC repo at {self.repo_path}")
        except Exception:
            self.dvc_repo = DvcRepo.init(self.repo_path)
            logger.info(f"Initialized new DVC repo at {self.repo_path}")
        self.config = self._setup_remote(remote)
        logger.info(f"Configured DVC remote: {remote}")

    def _setup_remote(self, remote: str, remote_name: str = "myremote"):
        """
        Configure the DVC remote storage.

        Args:
            remote (str): URL of the DVC remote storage.
            remote_name (str, optional): Name to assign to the remote. Defaults to "myremote".

        Returns:
            Config: The DVC Config object after editing.
        """
        cfg = Config(self.dvc_repo.dvc_dir)
        with cfg.edit("repo") as conf:
            if remote_name not in conf.get("remote", {}):
                conf["remote"][remote_name] = {"url": remote}
            conf["core"]["remote"] = remote_name

        return cfg

    def add(self, path: str):
        """
        Add a file or directory to DVC tracking.

        Args:
            path (str): Relative path to the file or directory to add.
        """
        abs_path = str(self.repo_path / path)
        try:
            self.dvc_repo.scm_context.quiet = True
            self.dvc_repo.add(abs_path)
            self.dvc_repo.scm_context.quiet = False
            logger.info(f"Added {abs_path} to DVC tracking")
        except OutputAlreadyTrackedError:
            logger.info(f"{abs_path} is already tracked by DVC")

    def push(self):
        """
        Push tracked data to the configured DVC remote storage.
        """
        self.dvc_repo.push()
        logger.info("Pushed data to DVC remote")

    def pull(self):
        """
        Pull tracked data from the configured DVC remote storage.
        """
        self.dvc_repo.pull()
        logger.info("Pulled data from DVC remote")
