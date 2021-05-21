---
title: Unit Test Machine Learning Models with Label Studio
type: blog
order: 94
meta_title: Machine Learning Unit Testing with Label Studio
meta_description: Machine Learning Unit Testing with Label Studio 
---

Write no-code unit tests for your machine learning models with Label Studio, ensuring accuracy and reducing the likelihood of business case risk. 

This example uses the [detectron2](https://ai.facebook.com/blog/-detectron2-a-pytorch-based-modular-object-detection-library-/) object detection model based on PyTorch and implements a testing flow as a GitHub Action for CI integration, or as a python script to run using your own build and test tooling.

## Why to unit test machine learning models

It can be difficult to test machine learning models, as pointed out by [Jeremy Jordan](https://twitter.com/jeremyjordan)'s post on [Testing Machine Learning](https://www.jeremyjordan.me/testing-ml/) and [Angie Jone](https://twitter.com/techgirl1908)'s work on [Test Automation for Machine Learning](https://angiejones.tech/test-automation-for-machine-learning/). 

Most testing frameworks for machine learning evaluate the _performance_ of the model and neglect to identify possible bugs in the logic that the model learned during the definition or training process. This means you might risk deploying a model with critical logic failures. 


## How to test machine learning models

There are many ways to test machine learning models, but for unit testing, you want to start by identifying critical errors and the product requirements that the model needs to fulfill. 

### Identify critical errors

When you evaluate a new machine learning model, inspect the metrics and plots that summarize model performance with a validation dataset. This evaluation lets you compare performance between multiple models and make relative judgments, but isn't enough information to characterize specific model behaviors. 

For example, if you want to identify scenarios where the model consistently fails, you must do additional investigation. Usually, that investigation starts with a list of the most common egregious model errors with the validation dataset and manually categorizing those failures.


### Identify product requirements

A product manager or business analyst can manually define the qualified metrics that need to be achieved before a model can be considered successful and production-ready. It's difficult to specify these metrics in terms of accuracy or recall for a model, and is much easier to specify them as tangible use case examples. 

## How it works

In this example, unit test your model predictions by automatically comparing model predictions against a validation dataset of ground truth annotations that you define. 

<img src="/images/ml-test-blog/ML-unit-test-scheme.png" alt="Diagram showing Label Studio and Tensorflow combining with GitHub Actions to produce validated test results." class="gif-border" />

To start unit testing your machine learning models with Label Studio, do the following:

1. Define ground truth annotations with Label Studio.
2. Get ML model predictions for a dataset. 
3. Trigger GitHub Action to evaluate the model predictions against ground truth annotations.


## Create ground truth annotations

Upload test images, annotate them then export in raw JSON format `tasks.json`

## Get ML model predictions

Assume you can get raw output tensors from your model predictions. The crucial step here is to convert these tensor into Label Studio predictions.

You can do it manually by following [Label Studio guide]() or applying converter utility:

```python
predictions = BboxConverter(bboxes).from_task(task)
```

#### Example

Get Detectron2 model

```python
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

def get_model():
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file(MODEL))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL)
    return DefaultPredictor(cfg)
```

Then run object detector inference to produce `test_tasks.json` input for the next step

```python
import cv2
import json
from label_studio_converter import BboxConverter

def run_model():
    # taken from https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=Vk4gID50K03a
    model = get_model()
    # get tasks with annotations from Label Studio export
    tasks = json.load(open('tasks.json'))
    for task in tasks:
        image_path = task['data']['image']
        image = cv2.imread(image_path)
        outputs = model(image)
        bboxes = outputs['instances'].pred_boxes.cpu().detach().numpy()
        # create Label Studio predictions
        predictions = BboxConverter(bboxes).from_task(task)
        task['predictions'] = predictions

    # Save tasks with predictions for ML unit testing
    with open('test_tasks.json', mode='w') as fout:
        json.dump(tasks, fout)
```

## Run ML unit tests with Github Action

Add the following step to your [Github Action workflow]()

```yaml
  - name: Run Label Studio ML Unit tests
    uses: heartexlabs/label-studio-ml-test@master
    with:
      test_data: test_tasks.json
      m: mAP
      threshold: 0.9
```

Feel free to select different _metric_ functions to compare annotations, as well as error sensivity defined by _threshold_ parameter. 

#### Example

```yaml
name: ml-unit-test-example
on:
  push:
    branches: ['*', '*/*', master]

jobs:
  detectron2_unit_test:
    name: ML Unit Test with Detectron2
    runs-on: ubuntu-latest

    steps:
      # first your steps to get ML assets...

      - name: Run Label Studio ML Unit tests
        uses: heartexlabs/label-studio-ml-test@master
        with:
          test_data: test_tasks.json
```

## Run ML unit tests manually

If you don't want to rely on Github actions infrastructure, you can trigger Label Studio ML unit tests manually from any python environment.

Install testing framework:

```bash
pip install label-studio-ml-test
```

Then copy prepared `test_tasks.json` in into repo and run:

```bash
label-studio-ml-test --test_data test_tasks.json
```
