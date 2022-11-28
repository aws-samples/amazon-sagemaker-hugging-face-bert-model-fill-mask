## Deploy a BERT model from Hugging Face Model Hub to Amazon SageMaker for a Fill-Mask use case

This repository contains a sample that demonstrates how to deploy a BERT model from Hugging Face Model Hub to Amazon SageMaker for a Fill-Mask use case.

### Overview

[Amazon SageMaker](https://aws.amazon.com/sagemaker/) is a fully managed end-to-end Machine Learning (ML) service that lets you build, train and deploy ML models for any use case with a fully managed infrastructure, tools and workflows.  SageMaker has a feature that enables customers to train, fine-tune and run inference using [Hugging Face](https://huggingface.co/) models for [Natural Language Processing (NLP)](https://en.wikipedia.org/wiki/Natural_language_processing) on SageMaker.

[Hugging Face](https://huggingface.co/) is an AI community that provides popular open-source NLP libraries and models.  AWS and Hugging Face have a [partnership](https://huggingface.co/blog/the-partnership-amazon-sagemaker-and-hugging-face) that allows a seamless integration through SageMaker with a set of AWS Deep Learning Containers (DLCs) for training and inference in PyTorch or TensorFlow and Hugging Face estimators and predictors for the SageMaker Python SDK.  These capabilities in SageMaker help developers and data scientists get started with NLP on AWS more easily.

This example demonstrates how to use the [SageMaker Hugging Face Inference Toolkit](https://github.com/aws/sagemaker-huggingface-inference-toolkit) to deploy [bert-base-uncased](https://huggingface.co/bert-base-uncased) which is a pre-trained [BERT](https://huggingface.co/transformers/v3.0.2/model_doc/bert.html) model for a [Fill-Mask](https://huggingface.co/tasks/fill-mask) use case.

Note:

* This notebook should only be run from within a SageMaker notebook instance as it references SageMaker native APIs.
* At the time of writing this notebook, the most relevant latest version of the Jupyter notebook kernel for this notebook was `conda_python3` and this came built-in with SageMaker notebooks.
* This notebook uses CPU based instances for training.
* This notebook will create resources in the same AWS account and in the same region where this notebook is running.

### Repository structure

This repository contains

* [A Jupyter Notebook](https://github.com/aws-samples/amazon-sagemaker-hugging-face-bert-model-fill-mask/blob/main/notebooks/torch_hugging_face_bert_fill_mask.ipynb) to get started.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

