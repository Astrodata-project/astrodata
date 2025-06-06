import os
from git import Repo, GitCommandError, InvalidGitRepositoryError
from pathlib import Path
import logging
from functools import wraps

logger = logging.getLogger(__name__)


def git_operation(operation_name: str):
    """Decorator to handle Git operation errors gracefully."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except GitCommandError as e:
                logger.error(f"{operation_name} failed: {e}")
                return None
            except Exception as e:
                logger.error(f"Unexpected error in {operation_name}: {e}")
                return None

        return wrapper

    return decorator


class CodeTracker:
    def __init__(self, repo_path: str, ssh_key_path: str = None, token: str = None):
        self.repo_path = Path(repo_path).resolve()
        self.ssh_key_path = ssh_key_path
        self.token = token
        self.repo = self._initialize_repo()

    def _initialize_repo(self):
        """Initialize or open existing Git repository."""
        try:
            repo = Repo(self.repo_path)
            logger.info(f"Using existing Git repo at {self.repo_path}")
            return repo
        except (GitCommandError, InvalidGitRepositoryError):
            repo = Repo.init(self.repo_path)
            logger.info(f"Initialized new Git repo at {self.repo_path}")
            return repo

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

    def _has_commits(self) -> bool:
        """Check if repository has any commits."""
        return self.repo.head.is_valid()

    def _validate_paths(self, paths: list) -> list:
        """Convert and validate file paths."""
        existing_paths = []
        for path in paths:
            path_obj = Path(path)
            if not path_obj.is_absolute():
                path_obj = self.repo_path / path_obj

            if path_obj.exists():
                existing_paths.append(str(path_obj))
            else:
                logger.warning(f"Path does not exist: {path_obj}")
        return existing_paths

    @git_operation("add remote")
    def add_remote(self, name: str, url: str):
        """Add a remote if it doesn't already exist, and fetch/pull if repo is empty."""
        if name in self.repo.remotes:
            logger.info(f"Remote {name} already exists")
            return True

        self.repo.create_remote(name, url)
        logger.info(f"Added remote {name}: {url}")

        # If repo has no commits, try to fetch and pull from remote
        if not self._has_commits():
            self._fetch_and_pull(name)

        return True

    @git_operation("fetch and pull")
    def _fetch_and_pull(self, remote_name: str):
        """Fetch and pull from remote for empty repository."""
        remote = self.repo.remotes[remote_name]
        with self.repo.git.custom_environment(**self._git_env()):
            remote.fetch()
            logger.info(f"Fetched from remote {remote_name}")

            # Try to pull from default branch (main or master)
            for branch in ["main", "master"]:
                if self._try_pull(remote_name, branch):
                    break

    def _try_pull(self, remote_name: str, branch: str) -> bool:
        """Try to pull from a specific branch."""
        try:
            self.repo.git.pull(remote_name, branch)
            logger.info(f"Pulled from {remote_name}/{branch}")
            return True
        except GitCommandError:
            return False

    def track(
        self,
        paths: list,
        commit_message: str,
        remote_name: str = "origin",
        branch: str = "main",
    ):
        """Track specified paths by adding, committing, and pushing them."""
        existing_paths = self._validate_paths(paths)
        if not existing_paths:
            logger.warning("No valid paths to track")
            return False

        if not self._add_to_index(existing_paths):
            return False

        if not self._has_changes():
            logger.info("No changes to commit")
            return True

        commit = self._create_commit(commit_message)
        if not commit:
            return False

        return self._push_to_remote(remote_name, branch)

    @git_operation("add to index")
    def _add_to_index(self, paths: list) -> bool:
        """Add paths to Git index."""
        self.repo.index.add(paths)
        logger.info(f"Added {len(paths)} path(s) to index")
        return True

    def _has_changes(self) -> bool:
        """Check if there are changes to commit."""
        return bool(self.repo.index.diff("HEAD"))

    @git_operation("create commit")
    def _create_commit(self, message: str):
        """Create a commit with the given message."""
        commit = self.repo.index.commit(message)
        logger.info(f"Created commit: {commit.hexsha[:8]} - {message}")
        return commit

    @git_operation("push to remote")
    def _push_to_remote(self, remote_name: str, branch: str) -> bool:
        """Push current branch to remote."""
        if remote_name not in self.repo.remotes:
            logger.error(f"Remote '{remote_name}' does not exist")
            return False

        if not self._has_commits():
            logger.warning("Repository has no commits yet")
            return False

        self._ensure_branch(branch)

        remote = self.repo.remotes[remote_name]
        with self.repo.git.custom_environment(**self._git_env()):
            try:
                remote.push(branch)
                logger.info(f"Pushed to {remote_name}/{branch}")
            except GitCommandError as e:
                if "does not match any" in str(e):
                    self.repo.git.push("--set-upstream", remote_name, branch)
                    logger.info(f"Pushed new branch to {remote_name}/{branch}")
                else:
                    raise
        return True

    def _ensure_branch(self, branch: str):
        """Ensure we're on the correct branch."""
        current_branch = self.repo.active_branch.name
        if current_branch == branch:
            return

        if branch in [b.name for b in self.repo.heads]:
            self.repo.heads[branch].checkout()
            logger.info(f"Switched to existing branch: {branch}")
        else:
            new_branch = self.repo.create_head(branch)
            new_branch.checkout()
            logger.info(f"Created and switched to new branch: {branch}")
