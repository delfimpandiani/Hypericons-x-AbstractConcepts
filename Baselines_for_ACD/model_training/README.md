# Baselines for Abstract Concept Detection on ARTstract dataset

This project includes scripts for model training, testing, and prediction.

## Usage

1. Set the desired values for the variables in the script.
2. Call the corresponding functions to perform the desired actions.

## Main Script

The main script is used for model training, testing, and prediction.

```python
import torch
from data_loader import data_prep
from model_training_es import load_pretrained_model, multi_class_classifier
from model_testing import load_trained_model, test_new_image, get_confusion_matrix

# Set the desired device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")
class_names = ["comfort", "danger", "death", "excitement", "fitness", "freedom", "power", "safety"]

# Defined functions and variables

# Example usage:
dataset_path = "../Local_structured_dataset"
variables = ["overall_b", "resnet50", 3, "pretrained_multi_class", True]

# Train the model
trained_model, trained_model_PATH = multi_class_classifier(data_prep_type, train_loader, val_loader, test_loader, base_model_name, base_model, experiment_name, epochs, augmented_data)
trained_model.to(device)

# Test the model
trained_model = load_trained_model(experiment_name, base_model_name)
trained_model.to(device)
get_confusion_matrix(experiment_name, trained_model, test_loader, class_names)

# Perform model prediction on a new image
image_path = "goya.jpg"
test_new_image(experiment_name, trained_model, image_path)
```
## Execution with Different Combinations
To execute with many different combinations of models, data preps, epochs, etc., you can use the variable_patterns list in the script. Uncomment the desired combinations and execute the script.

```python
variable_patterns = [
    # [data_prep_type, base_model_name, epochs, model_arch, augmented_data]
    # ["ub", "vgg16", 20, "pretrained_multi_class", True],
    # ["ub", "vgg16", 20, "pretrained_multi_class", False],
    # ...
]

for variables in variable_patterns:
    train_model(variables)
    test_model(variables, class_names)
    model_prediction(variables, image_path)
```

# Extra: in more detail

## Dataset Prep and Splitting Script

The `data_loader.py` script is used to split a given dataset into three parts: train, validation, and test datasets. The splitting process can be customized based on the data preparation type and whether augmented data is used.

### Usage

- Specify the `data_prep_type` parameter to control the dataset splitting strategy.
- Set `augmented_data` parameter to `True` if augmented training data is desired.

### Functions

#### data_prep(dataset_path, data_prep_type, augmented_data)

Prepare the data for training and evaluation.

**Parameters:**
- `dataset_path` (str): Local path to the directory that contains the images, separated in folders with labels as names.
- `data_prep_type` (str): Type of data preparation. Possible values: "ub" (unbalanced classes), "overall_b" (overall balanced), "partition_b" (partition balanced).
- `augmented_data` (bool): Flag to indicate whether augmented training data should be used.

**Returns:**
- `train_loader` (DataLoader): DataLoader for the training dataset.
- `val_loader` (DataLoader): DataLoader for the validation dataset.
- `test_loader` (DataLoader): DataLoader for the test dataset.
- `train_dataset` (Subset): Subset of the training dataset.
- `val_dataset` (Subset): Subset of the validation dataset.
- `test_dataset` (Subset): Subset of the test dataset.

#### Example Usage

```python
import torch
import random
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split

# Set the random seed for reproducibility
seed = 123
torch.manual_seed(seed)

############## LOADING AND PREPPING DATA ################
def data_prep(dataset_path, data_prep_type, augmented_data):
    # ... function implementation ...
    pass

dataset_path = "../Local_structured_dataset"
train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = data_prep(dataset_path, "ub", augmented_data=True)
```

## Multi-Class Classifier for ACD Detection

The `model_training_es.py` script contains functions for training and evaluating a multi-class classifier model.

### Functions

#### load_pretrained_model(model_name)

Loads a pretrained model based on the specified `model_name`.

**Parameters:**
- `model_name` (str): Name of the model to load. Supported options are: "vgg16", "resnet50", "vit", "mlp_mixer".

**Returns:**
- `torch.nn.Module`: Pretrained model instance.

**Raises:**
- `ValueError`: If the specified `model_name` is not supported.

#### document_experiment(data_prep_type, experiment_name, base_model_name, epochs, optimizer, test_loss, test_accuracy, test_f1_score, plot_file, augmented_data)

Documents the results of an experiment by writing the experiment details to a CSV file.

**Parameters:**
- `data_prep_type` (str): Type of data preparation performed for the experiment.
- `experiment_name` (str): Name of the experiment.
- `base_model_name` (str): Name of the base model used in the experiment.
- `epochs` (int): Number of epochs the model was trained for.
- `optimizer` (str): Name of the optimizer used for training.
- `test_loss` (float): Loss achieved on the test dataset.
- `test_accuracy` (float): Accuracy achieved on the test dataset.
- `test_f1_score` (float): F1 score achieved on the test dataset.
- `plot_file` (str): Path to the metrics image file.
- `augmented_data` (bool): Indicates whether augmented data was used in the experiment.

**Returns:**
- None

#### multi_class_classifier(data_prep_type, train_loader, val_loader, test_loader, base_model_name, model, experiment_name, epochs, augmented_data)

Trains a multi-class classifier model and evaluates it on the test dataset.

**Parameters:**
- `data_prep_type` (str): Type of data preparation performed for the experiment.
- `train_loader` (`torch.utils.data.DataLoader`): DataLoader for the training dataset.
- `val_loader` (`torch.utils.data.DataLoader`): DataLoader for the validation dataset.
- `test_loader` (`torch.utils.data.DataLoader`): DataLoader for the test dataset.
- `base_model_name` (str): Name of the base model used in the experiment.
- `model` (`torch.nn.Module`): Pretrained model instance.
- `experiment_name` (str): Name of the experiment.
- `epochs` (int): Number of epochs to train the model for.
- `augmented_data` (bool): Indicates whether augmented data was used in the experiment.

**Returns:**
- `tuple`: A tuple containing the trained multi-class classifier model and the path to the saved model.

### Usage

1. Import the required modules:
```python
import os
import csv
import timm
import torch
import datetime
import torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, accuracy_score
```

2. Load a pretrained model:

```python
model = load_pretrained_model("vgg16")
```

3. Load a pretrained model:

```python
train_loader = ...
val_loader = ...
test_loader = ...
model, model_path = multi_class_classifier("data_prep_type", train_loader, val_loader, test_loader, "base_model_name", model, "experiment_name", epochs, augmented_data)
```

4. Document the experiment results:

```python
document_experiment("data_prep_type", "experiment_name", "base
```

## Model Testing and Evaluation

This script provides functions for testing and evaluating a trained model. It includes functionalities such as loading a trained model, documenting testing metrics, generating a confusion matrix, and performing inference on unseen images.

### Functions

#### load_trained_model(experiment_name, base_model_name)

Load a trained model from a specified experiment.

**Parameters:**
- `experiment_name` (str): Name of the experiment.
- `base_model_name` (str): Name of the base model used in the experiment.

**Returns:**
- `model` (`torch.nn.Module`): Loaded trained model.

#### document_testing_metrics(experiment_name, test_loss, test_accuracy, test_f1_score, cm_path)

Document the testing metrics to a CSV file.

**Parameters:**
- `experiment_name` (str): Name of the experiment.
- `test_loss` (float): Testing loss.
- `test_accuracy` (float): Testing accuracy.
- `test_f1_score` (float): Testing F1 score.
- `cm_path` (str): Path to save the confusion matrix plot.

**Returns:**
- None

#### get_confusion_matrix(experiment_name, model, test_loader, class_names)

Generate the confusion matrix for the trained model and calculate testing metrics.

**Parameters:**
- `experiment_name` (str): Name of the experiment.
- `model` (`torch.nn.Module`): Trained model.
- `test_loader` (`torch.utils.data.DataLoader`): DataLoader for the test dataset.
- `class_names` (list): List of class names.

**Returns:**
- `cm` (numpy.ndarray): Confusion matrix.

#### test_new_image(experiment_name, model, image_path)

Perform inference on a new image using a given model.

**Parameters:**
- `experiment_name` (str): The name of the experiment or model being used.
- `model` (`torch.nn.Module`): The pre-trained model to use for inference.
- `image_path` (str): The path to the image file to be tested.

**Returns:**
- None

