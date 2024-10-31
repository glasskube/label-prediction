# label-prediction

Trying to make use of our [Kubeflow installation on GCP](https://glasskube.dev/blog/kubeflow-setup-guide), 
we wanted to create a label predictor for new GitHub issues in our [glasskube/glasskube](https://github.com/glasskube/glasskube) repo.

It doesn't really work. Turns out 300 issues aren't enough data, plus ML skills would have been beneficiary. Anyway here is what we did.

## Setup

This requires a working [Kubeflow installation](https://glasskube.dev/blog/kubeflow-setup-guide).

As we are far from being ML experts, we based our experiments on these excellent resources:
* [How to Fine-Tune LLMs with Kubeflow](https://www.kubeflow.org/docs/components/training/user-guides/fine-tuning/)
* [Fine-Tune BERT LLM for Sentiment Analysis with Kubeflow PyTorchJob (Jupyter Notebook)](https://github.com/kubeflow/training-operator/blob/master/examples/pytorch/text-classification/Fine-Tune-BERT-LLM.ipynb)
* [Fine-tuning BERT (and friends) for multi-label text classification](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb)

We combined these to perform our LLM finetuning for multi-label classification with kubeflow, with the following adaptions:
* We use our own dataset containing the github issues of our repo.
* Our custom dataset will be loaded from a GCS bucket instead of huggingface, and we publish the finetuned model back to a GCS bucket.

## Preparing the dataset

We use the Github CLI `gh` to generate the dataset:

```shell
gh issue list --state all --limit 1000 \
  --json title,body,labels \
  --jq 'map(del(.labels) + (.labels | map( { (.name|tostring): true } ) | add))' \
  > prepared-issues.json
```

The `--jq` part maps the string labels to be true-valued booleans in each JSON object. This works for any Github repo.
It might make sense to take only a subset of labels into consideration, like in our case `enhancement` and `bug`, e.g. like this:

```shell
jq 'map({"title": .title, "body": .body, "enhancement": (.enhancement // false), "bug": (.bug // false)})' prepared-issues.json > prepared-issues-reduced.json
```

Consequently a GitHub issue will be represented the following way:

```json
{
  "title": "Support transitive dependencies",
  "body": "...",
  "enhancement": true,
  "bug": false
}
```

A bucket inside the Google cloud storage of your project can be used to store this prepared dataset, from where it can be accessed by the training code afterwards.

## Kubeflow Notebooks

The provided Jupyter Notebooks can be opened in [Kubeflow Notebooks](https://www.kubeflow.org/docs/components/notebooks/overview/). 
You first need to create a Notebook and provide a name for it. The defaults given for the resource requests and limits should be sufficient. 

After connecting to the notebook, files can be created or uploaded – as a first test you can upload the `fine_tuning_kubeflow.ipynb` and run the
installation commands to ensure that all dependencies are present on the notebook server. 

Note that in this setup, the files you are working with are stored on a volume in the cluster. Make sure to download the files after editing, otherwise
your progress will be lost when you delete the kubeflow instance again. There surely are more sophisticated ways like connecting to a git repo
directly, but we didn't try this.

### Distributed training across GPU nodes

The `fine_tuning_kubeflow.ipynb` guides you through the setup necessary for distributed training.

## Automated finetuning with Kubeflow Pipelines

With [Kubeflow Pipelines](https://www.kubeflow.org/docs/components/pipelines/overview/), tasks like the previously shown finetuning can be automated without the user having to manually execute commands via the notebook.
This might be the preferred way once the training part has been figured out. After all, one might want to run finetuning again with an updated dataset in the future.

After making ourselves familiar with the concept of KFP and running a first [hello world pipeline](https://www.kubeflow.org/docs/components/pipelines/getting-started/)
we migrated the previously shown Jupyter Notebook to a Kubeflow pipeline with one component.

```python
from kfp import dsl, compiler

@dsl.component(packages_to_install=[
    "gcsfs",
    "transformers",
    "datasets==2.16",
    "evaluate==0.4.3",
    "accelerate",
    "scikit-learn",
    "kubeflow-training"
])
def start_distributed_training(
    bucket: str,
    dataset_file: str,
    output_model_name: str,
    gproject: str) -> str:

    # imports ...

    def train_func(parameters):
        # imports again ...
        # train_func like in the notebook

    job_name = "training-pipeline-job"
    # Create PyTorchJob
    # ...
    # Wait until PyTorchJob has Running condition.
    job = TrainingClient().wait_for_job_conditions(
        job_name,
        expected_conditions={"Running"},
    )
    return "job is running"

@dsl.pipeline
def training_pipeline(
    bucket: str,
    dataset_file: str,
    output_model_name: str,
    gproject: str
) -> str:
    training_task = start_distributed_training(
        bucket=bucket,
        dataset_file=dataset_file,
        output_model_name=output_model_name,
        gproject=gproject
    )
    return training_task.output

compiler.Compiler().compile(training_pipeline, 'training_pipeline.yaml')
```

Take a look at the `fine_tuning_pipeline.ipynb` notebook for the complete example. 

Running this script will compile the pipeline into the `training_pipeline.yaml` file, which can be uploaded and used in Kubeflow.
There might be better ways to deal with this in a more integrated manner.

The pipeline can be started in the Kubeflow UI, which is the point at which the parameters have to be defined. In our case these are the GCS bucket name,
the filename of the dataset, the name of the finetuned model, and the ID of the google cloud project. These are passed from the UI down
to the Pipeline, and from there on to the component and the training function.

