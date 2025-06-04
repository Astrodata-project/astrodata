import os
from git import Repo, GitCommandError, InvalidGitRepositoryError
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CodeTracker:
    def __init__(self, repo_path: str, ssh_key_path: str = None, token: str = None):
        self.repo_path = Path(repo_path).resolve()
        self.ssh_key_path = ssh_key_path
        self.token = token
        try:
            self.repo = Repo(self.repo_path)
            logger.info(f"Using existing Git repo at {self.repo_path}")
        except (GitCommandError, InvalidGitRepositoryError):
            self.repo = Repo.init(self.repo_path)
            logger.info(f"Initialized new Git repo at {self.repo_path}")

    def _git_env(self):
        """Set up environment variables for Git authentication."""
        env = os.environ.copy()
        if self.ssh_key_path:
            env["GIT_SSH_COMMAND"] = f"ssh -i {self.ssh_key_path}"
        if self.token:
            env["GIT_ASKPASS"] = "echo"
            env["GIT_USERNAME"] = "x-access-token"
            env["GIT_PASSWORD"] = self.token
        return env

    def add_and_commit(self, paths, message: str):
        if isinstance(paths, str):
            paths = [paths]
        paths = [str(self.repo_path / Path(p)) for p in paths]
        self.repo.index.add(paths)
        self.repo.index.commit(message)
        logger.info(f"Committed: {message}")

    def add_remote(self, name: str, url: str):
        if name not in self.repo.remotes:
            origin = self.repo.create_remote(name, url)
            logger.info(f"Added remote {name}: {url}")
        else:
            logger.warning(f"Remote {name} already exists")

    def push(self, remote: str = "origin", branch: str = "main"):
        if not self.repo.head.is_valid():
            # Write the repository name to README.md
            readme_path = self.repo_path / "README.md"
            repo_name = self.repo_path.name
            with open(readme_path, "w") as readme_file:
                readme_file.write(
                    f"# {repo_name}\n\nThis repository is managed by astrodata."
                )

            # Add and commit the README.md
            self.add_and_commit(["README.md"], "Add README.md with repository name")
            logger.info("Created initial commit with README.md")

            with self.repo.git.custom_environment(**self._git_env()):
                self.repo.git.push("--set-upstream", remote, branch)

        origin = self.repo.remotes[remote]
        with self.repo.git.custom_environment(**self._git_env()):
            origin.push(branch)
        logger.info(f"Pushed to {remote}/{branch}")
