import logging
from pathlib import Path

from dvc.config import Config
from dvc.output import OutputAlreadyTrackedError
from dvc.repo import Repo as DvcRepo

logger = logging.getLogger(__name__)


class DataTracker:
    def __init__(self, repo_path: str, remote: str):
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
        cfg = Config(self.dvc_repo.dvc_dir)
        with cfg.edit("repo") as conf:
            if remote_name not in conf.get("remote", {}):
                conf["remote"][remote_name] = {"url": remote}
            conf["core"]["remote"] = remote_name

        return cfg

    def add(self, path: str):
        abs_path = str(self.repo_path / path)
        try:
            self.dvc_repo.add(abs_path)
            logger.info(f"Added {abs_path} to DVC tracking")
        except OutputAlreadyTrackedError:
            logger.info(f"{abs_path} is already tracked by DVC")

    def push(self):
        self.dvc_repo.push()
        logger.info("Pushed data to DVC remote")

    def pull(self):
        self.dvc_repo.pull()
        logger.info("Pulled data from DVC remote")
