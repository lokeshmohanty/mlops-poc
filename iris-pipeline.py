from clearml.automation.controller import PipelineDecorator
from clearml import TaskTypes

# Make the following function an independent pipeline component step
# notice all package imports inside the function will be automatically logged as
# required packages for the pipeline execution step
@PipelineDecorator.component(return_values=["data_frame"], cache=True, task_type=TaskTypes.data_processing)
def get_data(pickle_data_url: str, extra: int = 43):
    print("Dataset")
    import sklearn  # noqa
    import pickle
    import pandas as pd
    from clearml import StorageManager

    local_iris_pkl = StorageManager.get_local_copy(remote_url=pickle_data_url)
    with open(local_iris_pkl, "rb") as f:
        iris = pickle.load(f)
    data_frame = pd.DataFrame(iris["data"], columns=iris["feature_names"])
    data_frame.columns += ["target"]
    data_frame["target"] = iris["target"]
    return data_frame


# Specifying `return_values` makes sure the function step can return an object to the pipeline logic
# In this case, the returned tuple will be stored as an artifact named "X_train, X_test, y_train, y_test"
@PipelineDecorator.component(
    return_values=["X_train", "X_test", "y_train", "y_test"], cache=True, task_type=TaskTypes.data_processing
)
def process_data(data_frame, test_size=0.2, random_state=42):
    print("step_two")
    import pandas as pd
    from sklearn.model_selection import train_test_split

    y = data_frame["target"]
    X = data_frame[(c for c in data_frame.columns if c != "target")]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test


@PipelineDecorator.component(return_values=["model"], cache=True, task_type=TaskTypes.training)
def train_model(X_train, y_train):
    print("step_three")
    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(solver="liblinear", multi_class="auto")
    model.fit(X_train, y_train)
    return model


@PipelineDecorator.component(return_values=["accuracy"], cache=True, task_type=TaskTypes.qc)
def eval_model(model, X_data, Y_data):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    Y_pred = model.predict(X_data)
    return accuracy_score(Y_data, Y_pred, normalize=True)


# The actual pipeline execution context
# notice that all pipeline component function calls are actually executed remotely
# Only when a return value is used, the pipeline logic will wait for the component execution to complete
@PipelineDecorator.pipeline(name="Iris Classification Pipeline", project="Iris Classification", version="1.0.0")
def execute_pipeline(pickle_url, mock_parameter="mock"):
    print("pipeline args:", pickle_url, mock_parameter)

    data_frame = get_data(pickle_url)
    X_train, X_test, y_train, y_test = process_data(data_frame)
    model = train_model(X_train, y_train)

    # Notice since we are "printing" the `model` object,
    # we actually deserialize the object from the third step, and thus wait for the third step to complete.
    print("returned model: {}".format(model))
    accuracy = 100 * eval_model(model, X_test, y_test)

    # Notice since we are "printing" the `accuracy` object,
    # we actually deserialize the object from the fourth step, and thus wait for the fourth step to complete.
    print(f"Accuracy={accuracy}%")


if __name__ == "__main__":
    # set the pipeline steps default execution queue (per specific step we can override it with the decorator)
    # PipelineDecorator.set_default_execution_queue('default')
    # Run the pipeline steps as subprocesses on the current machine, great for local executions
    # (for easy development / debugging, use `PipelineDecorator.debug_pipeline()` to execute steps as regular functions)
    PipelineDecorator.run_locally()

    # Start the pipeline execution logic.
    execute_pipeline(
        pickle_url="https://github.com/allegroai/events/raw/master/odsc20-east/generic/iris_dataset.pkl",
    )

    print("process completed")
