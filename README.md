# MLOps with ClearML

## Setup

```sh
  pip install clearml
  clearml-init
```

### Log experiments

```python
  from clearml import Task
  task = Task.init(project_name='great project', task_name='best experiment')
```

## Agent

```sh
  pip install clearml-agent
  clearml-agent init
```

### Start the agent's daemon and add it to the queue

```sh
  clearml-agent daemon --queue default
```


## Dataset ((clearml-data)[https://clear.ml/docs/latest/docs/clearml_data/data_management_examples/workflows])

### Create from local data

```sh
  # Create an empty dataset
  clearml-data crate --project <project name> --name <dataset-name>
  
  # Add files and folders to the dataset
  clearml-data add --files <file/folder>
  
  # Compress and upload the added files and folders
  clearml-data close
  
  # List dataset contents
  clearml-data --name <dataset-name>
```


### Modifying an existing dataset

An existing dataset can be modified by creating a child dataset and then making the
required changes using `clearml-data add` and `clearml-data remove`. To create a child dataset, pass an extra param `--parents <id>` while creating the dataset.
