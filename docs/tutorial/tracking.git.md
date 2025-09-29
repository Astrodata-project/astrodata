# git

## Code Versioning Setup

To enable code versioning with AstroData, you need to properly configure your YAML file. Follow these steps:

1. **Create a Remote Repository**  
   Set up a remote Git repository (e.g., on GitHub, GitLab, etc.).

2. **Configure the Remote URL**  
   In your YAML config file, set the `code.remote.url` field to the link of your remote repository.

3. **Set the Default Branch**  
   Specify the branch you want to use for versioning with the `code.branch` field.

4. **Select Files/Folders to Track**  
   List the files or folders you want to version under `code.tracked_files`.

5. **(Optional) Define Commit Message**  
   You can set a default commit message in the YAML file with `code.commit_message`.  
   Alternatively, you can provide a commit message directly in your code by calling `tracker.track(commit_message)`.  
   If both are provided, the message passed to `tracker.track()` takes priority.
  ```python
  from astrodata.tracking import Tracker

  # Initialize the tracker with your YAML config file
  tracker = Tracker(config_path="config.yaml")

  # Track changes with a custom commit message
  tracker.track("Refactor preprocessing pipeline")
  ```

Once your configuration is set, you can start working and use the Tracker to manage your code versioning automatically.
```{warning}
If there is any conflict (e.g., merge conflict or push failure), the tool will not perform any operation and will leave your files unchanged. You will need to resolve the conflict manually before proceeding with further tracking operations.
```
