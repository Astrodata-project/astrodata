from dvc.repo import Repo as DvcRepo
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DataTracker:
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path).resolve()
        try:
            self.dvc_repo = DvcRepo(self.repo_path)
            logger.info(f"Using existing DVC repo at {self.repo_path}")
        except Exception:
            self.dvc_repo = DvcRepo.init(self.repo_path)
            logger.info(f"Initialized new DVC repo at {self.repo_path}")

    def track(self, path: str):
        abs_path = str(self.repo_path / path)
        self.dvc_repo.add(abs_path)
        logger.info(f"Added {abs_path} to DVC tracking")

    def commit(self):
        self.dvc_repo.scm.add([".dvc", ".gitignore"])
        self.dvc_repo.scm.commit("Track data with DVC")
        logger.info("Committed DVC tracking changes")

    def push(self):
        self.dvc_repo.push()
        logger.info("Pushed data to DVC remote")
