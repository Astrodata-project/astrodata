import logging
from pathlib import Path

from astrodata.utils.utils import read_config

from .code_tracking import CodeTracker
from .data_tracking import DataTracker

logger = logging.getLogger(__name__)


class Tracker:
    def __init__(self, config_path: str):
        self.config = read_config(config_path)
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

    def track(self):
        print(f"Tracking code in {self.project_path}")
        self._track_code()
        print("Code tracking completed.")
        self._track_data()

    def _track_code(self):
        if not self.code_tracker:
            return

        code_config = self.config.get("code", {})

        # Add remote if configured
        if code_config.get("remote", {}).get("enable", False):
            remote = code_config["remote"]
            self.code_tracker.add_remote(remote["name"], remote["url"])

        # Track files if auto_commit is enabled
        if code_config.get("auto_commit", False):
            tracked_files = code_config.get("tracked_files", ["src", "pyproject.toml"])
            msg = code_config.get("commit_message", "Auto commit by astrodata")
            remote_name = code_config.get("remote", {}).get("name", "origin")
            branch = code_config.get("branch", "main")

            self.code_tracker.track(tracked_files, msg, remote_name, branch)

    def _track_data(self):
        if not self.data_tracker:
            return

        data_config = self.config.get("data", {})
        for path in data_config.get("paths", []):
            self.data_tracker.track(path)
        self.data_tracker.commit()

        if data_config.get("push", False):
            self.data_tracker.push()
