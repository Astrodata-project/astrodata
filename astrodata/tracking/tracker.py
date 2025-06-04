import yaml
from pathlib import Path
import logging

from .code_tracking import CodeTracker
from .data_tracking import DataTracker

logger = logging.getLogger(__name__)


class Tracker:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.project_path = Path(self.config["project_path"]).resolve()

        self.code_tracker = None
        self.data_tracker = None

        if self.config.get("code", {}).get("enable", False):
            ssh_key = self.config.get("code", {}).get("ssh_key_path")
            token = self.config.get("code", {}).get("token")
            self.code_tracker = CodeTracker(
                self.project_path, ssh_key_path=ssh_key, token=token
            )
        if self.config.get("data", {}).get("enable", False):
            self.data_tracker = DataTracker(self.project_path)

    def _load_config(self, path):
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def track(self):
        self._track_code()
        self._track_data()

    def _track_code(self):
        if not self.code_tracker:
            return

        code_config = self.config.get("code", {})
        if code_config.get("auto_commit", False):
            tracked_files = code_config.get("tracked_files", ["src", "pyproject.toml"])
            msg = code_config.get("commit_message", "Auto commit by astrodata")
            self.code_tracker.add_and_commit(tracked_files, msg)

        if code_config.get("remote", {}).get("enable", False):
            remote = code_config["remote"]
            self.code_tracker.add_remote(remote["name"], remote["url"])
            self.code_tracker.push(remote["name"])

    def _track_data(self):
        if not self.data_tracker:
            return

        data_config = self.config.get("data", {})
        for path in data_config.get("paths", []):
            self.data_tracker.track(path)
        self.data_tracker.commit()

        if data_config.get("push", False):
            self.data_tracker.push()
