import os
from functools import wraps
from pathlib import Path

from git import GitCommandError, InvalidGitRepositoryError, Repo

from astrodata.utils.logger import setup_logger

logger = setup_logger(__name__)


def git_operation(operation_name: str):
    """
    Decorator to handle Git operation errors gracefully.

    Args:
        operation_name (str): The name of the Git operation for logging and error messages.

    Returns:
        Callable: A decorator that wraps the function and handles Git errors.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except GitCommandError as e:
                if operation_name == "prune and align branches":
                    logger.warning(
                        f"{operation_name} failed: {e}. This may be due to no remote branches."
                    )
                    return None
                raise RuntimeError(
                    f"Git operation '{operation_name}' failed: {e}"
                ) from e
            except Exception as e:
                logger.error(f"Unexpected error in {operation_name}: {e}")
                return None

        return wrapper

    return decorator


class CodeTracker:
    """
    A class to manage and track code changes in a Git repository.

    This class provides methods for common Git operations such as branch management,
    remote handling, committing, pushing, and synchronizing with remotes.

    Args:
        repo_path (str): Path to the local Git repository.
        ssh_key_path (str, optional): Path to the SSH private key for authentication.
        token (str, optional): Personal access token for HTTPS authentication.
        branch (str, optional): Default branch to use. Defaults to "main".
    """

    def __init__(
        self,
        repo_path: str,
        ssh_key_path: str = None,
        token: str = None,
        branch: str = "main",
    ):
        self.repo_path = Path(repo_path).resolve()
        self.ssh_key_path = ssh_key_path
        self.token = token
        self.repo = self._initialize_repo()
        self.branch = branch

    def _initialize_repo(self):
        """
        Initialize or open an existing Git repository.

        Returns:
            Repo: An instance of the GitPython Repo object.
        """
        try:
            repo = Repo(self.repo_path)
            logger.info(f"Using existing Git repo at {self.repo_path}")
            return repo
        except (GitCommandError, InvalidGitRepositoryError):
            repo = Repo.init(self.repo_path)
            logger.info(f"Initialized new Git repo at {self.repo_path}")
            return repo

    def _git_env(self):
        """
        Set up environment variables for Git authentication.

        Returns:
            dict: A copy of the environment variables with Git authentication settings.
        """
        env = os.environ.copy()
        if self.ssh_key_path:
            env["GIT_SSH_COMMAND"] = f"ssh -i {self.ssh_key_path}"
        if self.token:
            env["GIT_ASKPASS"] = "echo"
            env["GIT_USERNAME"] = "x-access-token"
            env["GIT_PASSWORD"] = self.token
        return env

    def _has_commits(self) -> bool:
        """
        Check if the repository has any commits.

        Returns:
            bool: True if the repository has at least one commit, False otherwise.
        """
        return self.repo.head.is_valid()

    def _validate_paths(self, paths: list) -> list:
        """
        Convert and validate file paths, ensuring they exist.

        Args:
            paths (list): List of file or directory paths.

        Returns:
            list: List of existing absolute paths as strings.
        """
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

    @git_operation("checkout")
    def checkout(self, branch_name: str):
        """
        Checkout to the specified branch.

        If the branch does not exist, create it from HEAD.

        Args:
            branch_name (str): The name of the branch to checkout.

        Returns:
            bool: True if checked out to an existing branch, False if a new branch was created.
        """
        if branch_name in [b.name for b in self.repo.heads]:
            self.repo.heads[branch_name].checkout()
            self.branch = branch_name
            logger.info(f"Checked out to existing branch '{branch_name}'")
            return True
        else:
            self._create_branch(branch_name)

        return False

    @git_operation("create branch")
    def _create_branch(self, branch_name: str, base: str = "HEAD") -> bool:
        """
        Create a new branch from the given base.

        Args:
            branch_name (str): The name of the new branch.
            base (str, optional): The base commit or branch. Defaults to "HEAD".

        Returns:
            bool: True if the branch was created successfully.
        """
        self.repo.create_head(branch_name, base)
        self.repo.heads[branch_name].checkout()
        self.branch = branch_name
        logger.info(f"Created branch '{branch_name}' from '{base}'")
        self.repo.git.push("--set-upstream", "origin", branch_name)
        return True

    @git_operation("add remote")
    def add_remote(self, name: str, url: str):
        """
        Add a remote if it doesn't already exist, and fetch/pull if the repo is empty.

        Args:
            name (str): The name of the remote.
            url (str): The URL of the remote repository.

        Returns:
            bool: True if the remote was added or already exists.
        """
        if name in self.repo.remotes:
            logger.info(f"Remote {name} already exists")
            return True

        self.repo.create_remote(name, url)
        logger.info(f"Added remote {name}: {url}")

        # Always fetch after adding remote
        self.repo.remotes[name].fetch()
        logger.info(f"Fetched from remote {name}")

        # If repo has no commits, try to pull from remote
        if not self._has_commits():
            # Try to pull default branch if possible
            try:
                self.pull(name, self.branch)
            except Exception as e:
                logger.warning(f"Could not pull after adding remote: {e}")

        return True

    @git_operation("pull")
    def pull(self, remote_name: str, branch: str):
        """
        Fetch and pull from remote for empty repository.

        Args:
            remote_name (str): The name of the remote.
            branch (str): The branch to pull.

        Returns:
            bool: True if pull was successful, None otherwise.
        """
        remote = self.repo.remotes[remote_name]
        with self.repo.git.custom_environment(**self._git_env()):
            remote.fetch()
            logger.info(f"Fetched from remote {remote_name}")

            if self._try_pull(remote_name, branch):
                return True

    @git_operation("reset merge")
    def _reset_merge(self):
        """
        Abort a merge in progress (used to recover from conflicts).

        Returns:
            None
        """
        self.repo.git.merge("--abort")
        logger.info("Merge aborted due to conflict")

    def _try_pull(self, remote_name: str, branch: str) -> bool:
        """
        Try to pull from a specific branch. Abort if merge conflict occurs.

        Args:
            remote_name (str): The name of the remote.
            branch (str): The branch to pull.

        Returns:
            bool: True if pull was successful, False otherwise.
        """
        try:
            self.repo.git.pull(remote_name, branch)
            logger.info(f"Pulled from {remote_name}/{branch}")
            return True
        except GitCommandError as e:
            error_msg = str(e)
            if "would be overwritten by merge" in error_msg:
                logger.error(
                    f"Pull failed: Untracked files would be overwritten. "
                    f"Please move or commit these files, then retry. Details: {error_msg}"
                )
            elif "Merge conflict" in error_msg or "CONFLICT" in error_msg:
                logger.error(
                    f"Merge conflict detected during pull from {remote_name}/{branch}"
                )
                self._reset_merge()
            else:
                logger.error(f"Pull failed: {e}")

    def add_to_index(self, paths: list) -> bool:
        """
        Add paths to the Git index (staging area).

        Args:
            paths (list): List of file or directory paths to add.

        Returns:
            bool: True if files were added to the index.
        """
        # Convert to relative paths for ignored() check
        self.repo.index.add(paths)
        logger.info(f"Added {len(paths)} path(s) to index")
        return True

    @git_operation("remove from index")
    def remove_deleted_from_index(self) -> bool:
        """
        Remove deleted files from the Git index.

        Returns:
            bool: True if deleted files were removed from the index.
        """
        paths = []
        for obj in self.repo.index.diff(None):
            if obj.change_type == "D":
                paths.append(obj.a_path)
        if paths != []:
            self.repo.index.remove(paths, r=True)
        logger.info(f"Removed {len(paths)} path(s) from index")
        return True

    def _has_changes(self) -> bool:
        """
        Check if there are changes to commit.

        Returns:
            bool: True if there are staged changes to commit, False otherwise.
        """
        if not self._has_commits():
            # If no commits yet, any staged files are changes
            return bool(self.repo.index.entries)
        return bool(self.repo.index.diff("HEAD"))

    @git_operation("commit")
    def create_commit(self, message: str):
        """
        Create a commit with the given message.

        Args:
            message (str): The commit message.

        Returns:
            Commit: The created commit object, or False if no changes to commit.
        """
        if not self._has_changes():
            logger.info("No changes to commit")
            return False
        commit = self.repo.index.commit(message)
        logger.info(f"Created commit: {commit.hexsha[:8]} - {message}")
        return commit

    @git_operation("push")
    def push(self, remote_name: str, branch: str, remote_url: str = None) -> bool:
        """
        Push the current branch to the specified remote.

        Args:
            remote_name (str): The name of the remote.
            branch (str): The branch to push.
            remote_url (str, optional): The URL of the remote, if it needs to be created.

        Returns:
            bool: True if push was successful, False otherwise.
        """
        if remote_name not in self.repo.remotes:
            if remote_url:
                self.repo.create_remote(remote_name, remote_url)
                logger.info(f"Added remote '{remote_name}': {remote_url}")
            else:
                logger.error(
                    f"Remote '{remote_name}' does not exist and no URL provided"
                )
                return False

        if not self._has_commits():
            logger.warning("Repository has no commits yet")
            return False

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

    @git_operation("prune and align branches")
    def align_with_remote(self, remote_name: str = "origin"):
        """
        Prune deleted remote branches and delete local branches whose remote is gone.

        Args:
            remote_name (str, optional): The name of the remote to align with. Defaults to "origin".

        Returns:
            None
        """
        # Prune remote-tracking branches
        self.repo.git.fetch("--prune", remote_name)
        logger.info(f"Pruned deleted branches (if any) from remote '{remote_name}'")

        gone_branches = [
            head.name
            for head in self.repo.heads
            if head.tracking_branch() and not head.tracking_branch().is_valid()
        ]
        for branch in gone_branches:
            self.repo.delete_head(branch, force=True)
            logger.info(f"Deleted local branch '{branch}' as its remote was deleted")
