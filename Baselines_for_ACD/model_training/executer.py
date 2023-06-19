"""
Main Script

This script is used for model training, testing, and prediction.

Usage:
    - Set the desired values for the variables.
    - Call the corresponding functions to perform the desired actions.
"""

import torch
from data_loader import data_prep
from model_training_es import load_pretrained_model, multi_class_classifier
from model_testing import load_trained_model, test_new_image, get_confusion_matrix

# ############# CHOOSE DEVICE ################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")
class_names = ["comfort", "danger", "death", "excitement", "fitness", "freedom", "power", "safety"]

def get_experiment_name (variables):

    data_prep_type, base_model_name, epochs, model_type, augmented_data = variables
    # specify experiment name
    augmented_variable = "augmented" if augmented_data else "not_augmented"
    experiment_name = str(
        data_prep_type + "_" + base_model_name + "_" + str(epochs) + "_epochs_" + model_type + "_" + augmented_variable)
    return experiment_name

def train_model(variables):
    data_prep_type, base_model_name, epochs, model_type, augmented_data = variables
    # specify experiment name
    experiment_name = get_experiment_name(variables)
    # Load data with desired variables
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = data_prep(dataset_path, data_prep_type, augmented_data=augmented_data, )
    # Load desired base model to the GPU
    base_model = load_pretrained_model(base_model_name)
    base_model.to(device)
    # train the model and load it on the GPU
    trained_model, trained_model_PATH = multi_class_classifier(data_prep_type, train_loader, val_loader,test_loader, base_model_name, base_model, experiment_name, epochs, augmented_data)
    trained_model.cuda()
    return trained_model, trained_model_PATH

def test_model(variables, class_names):
    data_prep_type, base_model_name, epochs, model_type, augmented_data = variables
    # specify experiment name
    experiment_name = get_experiment_name(variables)
    # Load data with desired balancing
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = data_prep(dataset_path, data_prep_type, augmented_data)
    trained_model = load_trained_model(experiment_name, base_model_name)
    trained_model.to(device)
    get_confusion_matrix(experiment_name, trained_model, test_loader, class_names)
    return

def model_prediction(variables, image_path, base_model_name):
    # specify experiment name
    experiment_name = get_experiment_name(variables)
    # load trained model
    trained_model = load_trained_model(experiment_name, base_model_name)
    # test new image
    test_new_image(experiment_name, trained_model, image_path)
    return

dataset_path = "../Local_structured_dataset"
variable = ["overall_b", "resnet50", 3, "pretrained_multi_class", True]
# variable = ["overall_b", "vgg16", 5, "pretrained_multi_class", True]
# variable = ["overall_b", "vit", 5, "pretrained_multi_class", True]
# train_model(variable)
# test_model(variable, class_names)
image_path = "goya.jpg"
model_prediction(variable, image_path, variable[1])



## To execute with many different combinations of models, data preps, epochs, etc:
# variable_patterns = [
## data_prep_type, base_model_name, epochs, model_arch, augmented_data
# ["ub", "vgg16", 20, "pretrained_multi_class", True],
# ["ub", "vgg16", 20, "pretrained_multi_class", False],
# ["ub", "resnet50", 20, "pretrained_multi_class", True],
# ["ub", "resnet50", 20, "pretrained_multi_class", False],
# ["overall_b", "vgg16", 100, "pretrained_multi_class", True],
# ["overall_b", "vgg16", 20, "pretrained_multi_class", False],
# ["overall_b", "resnet50", 20, "pretrained_multi_class", True],
# ["overall_b", "resnet50", 20, "pretrained_multi_class", False],
# ["partition_b", "vgg16", 20, "pretrained_multi_class", True],
# ["partition_b", "vgg16", 20, "pretrained_multi_class", False],
# ["partition_b", "resnet50", 20, "pretrained_multi_class", True],
# ["partition_b", "resnet50", 20, "pretrained_multi_class", False],
# ["ub", "vit", 20, "pretrained_multi_class", True],
# ["ub", "vit", 20, "pretrained_multi_class", False],
# ["ub", "mlp_mixer", 20, "pretrained_multi_class", True],
# ["ub", "mlp_mixer", 20, "pretrained_multi_class", False],
# ["overall_b", "vit", 20, "pretrained_multi_class", True],
# ["overall_b", "vit", 20, "pretrained_multi_class", True],
# # ["overall_b", "mlp_mixer", 20, "pretrained_multi_class", False],
# ["partition_b", "vit", 20, "pretrained_multi_class", True],
# ["partition_b", "vit", 20, "pretrained_multi_class", False],
# # ["partition_b", "mlp_mixer", 20, "pretrained_multi_class", True],
# # ["partition_b", "mlp_mixer", 20, "pretrained_multi_class", False],
# ["overall_b", "vit", 60, "pretrained_multi_class", True]
# ]

# for variables in variable_patterns:
#     train_model(variables)
#     test_model(variables, class_names)
#     model_prediction(variables, image_path)