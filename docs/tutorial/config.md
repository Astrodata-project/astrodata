
# Configuration Reference

Astrodata uses a YAML configuration file to control code tracking, data tracking, and machine learning preprocessing. Below is a sample configuration and a description of each section.

## Configuration Sections

### project_path

- Root path of your project. Used to resolve relative paths.

### code

- **enable:** Enable or disable code tracking with Git.
- **auto_commit:** Automatically commit tracked files.
- **tracked_files:** List of files or directories to track.
- **commit_message:** Default commit message.
- **branch:** Git branch to use.
- **remote:**  
  - **enable:** Enable pushing to a remote repository.
  - **name:** Name of the remote (e.g., "origin").
  - **url:** URL of the remote repository.
- **ssh_key_path:** Path to SSH key for authentication.
- **token:** Token for authentication.

### data

- **enable:** Enable or disable data tracking with DVC.
- **paths:** List of data directories or files to track.
- **remote:** Full path to the DVC remote storage.
- **push:** Whether to push data to the remote after tracking.

### preml

The `preml` section of the configuration file allows you to define parameters for each preprocessing processor used in your machine learning pipeline. Each processor block must be named exactly as the corresponding processor class (e.g., `TrainTestSplitter`, `OHE`, `MissingImputator`). The YAML keys and values under each processor correspond directly to the parameters of that class.

For example, the processor class `TrainTestSplitter` that takes parameters like `targets`, `test_size`, and `random_state`, your configuration should look like:

```yaml
preml:
  TrainTestSplitter:
    targets:
      - "duration"
    test_size: 0.2
    random_state: 42
    validation:
      enabled: false
      size: 0.1
```

This approach ensures that your pipeline can be configured flexibly, and that all processor parameters are clearly specified in the config file. If you define processors both in code and in the config file, the code definitions will take precedence.

## Usage

Pass the path to this YAML file as `config_path` to `Tracker`, `DataPipeline`, or `PremlPipeline` to control their behavior.
