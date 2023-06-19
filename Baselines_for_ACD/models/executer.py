import torch
from data_loading import data_prep
from model_training_es import load_pretrained_model, multi_class_classifier, finetuned_multi_class_classifier
from model_testing import load_trained_model, test_new_image, get_confusion_matrix

# ############# CHOOSE DEVICE ################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

def get_experiment_name (variables):
    data_prep_type, base_model_name, epochs, model_type, augmented_data = variables
    # specify experiment name
    if augmented_data:
        augmented_variable = "augmented"
    else:
        augmented_variable = "not_augmented"
    experiment_name = str(
        data_prep_type + "_" + base_model_name + "_" + str(epochs) + "_epochs_" + model_type + "_" + augmented_variable)
    return experiment_name

def train_model(variables):
    data_prep_type, base_model_name, epochs, model_type, augmented_data = variables
    # specify experiment name
    experiment_name = get_experiment_name(variables)
    # Load data with desired variables
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = data_prep(data_prep_type, augmented_data=augmented_data)
    # Load desired base model to the GPU
    base_model = load_pretrained_model(base_model_name)
    base_model.to(device)
    # specify the training type
    if model_type == "pretrained_multi_class":
        trained_model, trained_model_PATH = multi_class_classifier(data_prep_type, train_loader, val_loader,test_loader, base_model_name, base_model, experiment_name, epochs, augmented_data)
    elif model_type == "finetuned_multi_class":
        trained_model, trained_model_PATH = finetuned_multi_class_classifier(data_prep_type, train_loader, val_loader,test_loader, base_model_name, base_model, experiment_name, epochs, augmented_data)
    trained_model.cuda()
    return trained_model, trained_model_PATH

def test_model(variables, class_names):
    data_prep_type, base_model_name, epochs, model_type, augmented_data = variables
    # specify experiment name
    experiment_name = get_experiment_name(variables)
    # Load data with desired balancing
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = data_prep(data_prep_type, augmented_data)
    trained_model = load_trained_model(experiment_name)
    get_confusion_matrix(experiment_name, trained_model, test_loader, class_names)
    return

def model_prediction(variables, image_path):
    # specify experiment name
    experiment_name = get_experiment_name(variables)
    # load trained model
    trained_model = load_trained_model(experiment_name)
    # test new image
    test_new_image(experiment_name, trained_model, image_path)
    return



# data_prep_type, base_model_name, epochs, model_arch, augmented_data
# variable_patterns = [
# ["ub", "vgg16", 25, "pretrained_multi_class", True],
# ["ub", "vgg16", 25, "pretrained_multi_class", False],
# ["ub", "resnet50", 25, "pretrained_multi_class", True],
# ["ub", "resnet50", 25, "pretrained_multi_class", False],
# ["overall_b", "vgg16", 25, "pretrained_multi_class", True],
# ["overall_b", "vgg16", 25, "pretrained_multi_class", False],
# ["overall_b", "resnet50", 25, "pretrained_multi_class", True],
# ["overall_b", "resnet50", 25, "pretrained_multi_class", False],
# ["partition_b", "vgg16", 25, "pretrained_multi_class", True],
# ["partition_b", "vgg16", 25, "pretrained_multi_class", False],
# ["partition_b", "resnet50", 25, "pretrained_multi_class", True],
# ["partition_b", "resnet50", 25, "pretrained_multi_class", False],

### it doesnt make sense to do the below, bc the pretrained and the finetuned are exactly the same right now
# ["ub", "vgg16", 25, "finetuned_multi_class", True],
# ["ub", "vgg16", 25, "finetuned_multi_class", False],
# ["ub", "resnet50", 25, "finetuned_multi_class", True],
# ["ub", "resnet50", 25, "finetuned_multi_class", False],
# ["overall_b", "vgg16", 25, "finetuned_multi_class", True],
# ["overall_b", "vgg16", 25, "finetuned_multi_class", False],
# ["overall_b", "resnet50", 25, "finetuned_multi_class", True],
# ["overall_b", "resnet50", 25, "finetuned_multi_class", False],
# ["partition_b", "vgg16", 25, "finetuned_multi_class", True],
# ["partition_b", "vgg16", 25, "finetuned_multi_class", False],
# ["partition_b", "resnet50", 25, "finetuned_multi_class", True],
# ["partition_b", "resnet50", 25, "finetuned_multi_class", False],
# ]



variable_patterns = [
# ["ub", "vgg16", 20, "pretrained_multi_class", True],
# ["ub", "vgg16", 20, "pretrained_multi_class", False],
# ["ub", "resnet50", 20, "pretrained_multi_class", True],
# ["ub", "resnet50", 20, "pretrained_multi_class", False],
["overall_b", "vgg16", 100, "pretrained_multi_class", True],
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
# ["overall_b", "vit", 20, "pretrainedtingined_multi_class", True],
# # ["overall_b", "mlp_mixer", 20, "pretrained_multi_class", False],
# ["partition_b", "vit", 20, "pretrained_multi_class", True],
# ["partition_b", "vit", 20, "pretrained_multi_class", False],
# # ["partition_b", "mlp_mixer", 20, "pretrained_multi_class", True],
# # ["partition_b", "mlp_mixer", 20, "pretrained_multi_class", False],
#
# ["overall_b", "vit", 60, "pretrained_multi_class", True]
]
class_names = ["comfort", "danger", "death", "excitement", "fitness", "freedom", "power", "safety"]
image_path = "unseen_images/goya.png"
# for variables in variable_patterns:
#     train_model(variables)
#     test_model(variables, class_names)
#     # model_prediction(variables, image_path)

# paper_variable = ["overall_b", "resnet50", 100, "pretrained_multi_class", True]
paper_variable2 = ["overall_b", "vit", 1000, "pretrained_multi_class", True]
# train_model(paper_variable)
# test_model(paper_variable, class_names)
train_model(paper_variable2)
test_model(paper_variable2, class_names)