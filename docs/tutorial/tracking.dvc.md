# DVC

## What is DVC and Why Use It?

[DVC (Data Version Control)](https://dvc.org/) is a tool for versioning large data files, datasets, and files in general. In the context of AstroData, DVC helps you track and version your data outputs (such as processed datasets or intermediate files), ensuring that your experiments are reproducible and your data is synchronized with your code.

Using DVC functionalitites within AstroData allows you to:

- Version control your data alongside your code.
- Reproduce experiments by tracking exactly which data was used.
- Share data and results easily with collaborators.

## Configuring DVC in AstroData

To enable data versioning with DVC, you need to configure the relevant section in your YAML file:

1. **Enable Data Tracking**  
   Set `data.enable: true` in your config.

2. **Specify Data Paths**  
   List the files or folders you want to track under `data.paths`. For example, raw data files you input into your pipeline.
```{hint}
Files (artifacts and data) produced by AstroData will be automatically tracked if you have `dump_output` enabled in your pipeline configuration.
```

3. **Set the DVC Remote**  
   Provide the full path to your DVC remote storage with `data.remote`. This is where your data will be pushed.

4. **Enable Data Push**  
   Set `data.push: true` to automatically push tracked data to the remote after each run.

Example configuration:
```yaml
data:
  enable: true
  paths:
    - "files_to_version"
  remote: "/path/to/remote/" # Must be full path
  push: true
```

When you run your data or preml pipeline with AstroData and have `dump_output` enabled, the outputs will be saved and tracked by DVC according to this configuration.

> **Note:** DVC operations are handled automatically by AstroData when configured. You can still use DVC from the command line as usual.
