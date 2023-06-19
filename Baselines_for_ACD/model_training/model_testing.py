"""
Model Testing and Evaluation

This script provides functions for testing and evaluating a trained model. It includes functionalities such as loading a trained model, documenting testing metrics, generating a confusion matrix, and performing inference on unseen images.

"""

import torch
import csv
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, accuracy_score
from model_training_es import load_pretrained_model

device = "cuda"
# Set the random seed for reproducibility
seed = 123
torch.manual_seed(seed)

def load_trained_model(experiment_name, base_model_name):
    """
    Load a trained model from a specified experiment.

    Args:
        experiment_name (str): Name of the experiment.

    Returns:
        model (torch.nn.Module): Loaded trained model.
    """
    model = load_pretrained_model(base_model_name)
    model_path = '../Local_logs/saved_models/' + experiment_name + 'best_model.pth'
    num_classes = 8
    try:
        model.classifier[-1] = torch.nn.Linear(in_features=4096, out_features=num_classes)
    except:
        if hasattr(model, "fc"):
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        else:
            model.head = torch.nn.Linear(model.head.in_features, num_classes)
    model.load_state_dict(torch.load(model_path))
    return model

def document_testing_metrics(experiment_name, test_loss, test_accuracy, test_f1_score, cm_path):
    """
       Document the testing metrics to a CSV file.

       Args:
           experiment_name (str): Name of the experiment.
           test_loss (float): Testing loss.
           test_accuracy (float): Testing accuracy.
           test_f1_score (float): Testing F1 score.
           cm_path (str): Path to save the confusion matrix plot.
    """
    try:
        with open('testing_metrics.csv', 'r') as f:
            exists = True
    except FileNotFoundError:
        exists = False

    # If the file does not exist, create it and write the header row
    if not exists:
        with open('../Local_logs/test_stats/testing_metrics.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(
                ["experiment_name", "test_loss", "test_accuracy", "test_f1_score", "cm_path"])

    # Write the data for the current experiment
    with open('../Local_logs/test_stats/testing_metrics.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([experiment_name, test_loss, test_accuracy, test_f1_score, cm_path])
    return

def get_confusion_matrix(experiment_name, model, test_loader, class_names):
    """
    Generate the confusion matrix for the trained model and calculate testing metrics.

    Args:
        experiment_name (str): Name of the experiment.
        model (torch.nn.Module): Trained model.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        class_names (list): List of class names.

    Returns:
        cm (numpy.ndarray): Confusion matrix.
    """
    test_loss = 0
    test_accuracy = 0
    test_f1_score = 0
    criterion = torch.nn.CrossEntropyLoss()

    # test evaluation
    with torch.no_grad():
        model.eval()
        y_true = []
        y_pred = []
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.to(device))
            test_loss += loss.item()
            test_pred = torch.argmax(outputs, dim=1)
            test_accuracy += accuracy_score(labels.cpu().numpy(), test_pred.cpu().numpy())
            test_f1_score += f1_score(labels.cpu().numpy(), test_pred.cpu().numpy(),
                                                                average='weighted')
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.tolist())
            y_pred.extend(preds.tolist())

    # Get the test metrics
    test_loss /= len(test_loader)
    test_accuracy /= len(test_loader)
    test_f1_score /= len(test_loader)

    metric_statement = 'Test Loss: {:.4f} Test Accuracy: {:.4f} Test F1 Score: {:.4f}'.format(
        test_loss, test_accuracy, test_f1_score)
    print(metric_statement)

    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_path = '../Local_logs/metrics_plots/confusion_matrices/' + experiment_name
    # Plot the confusion matrix
    plt.figure(figsize=(10, 10))
    # sns.heatmap(cm, annot=True, cmap="Blues", fmt='g')
    sns.heatmap(cm, annot=True, cmap="Blues", fmt='g', xticklabels=class_names,
                yticklabels=class_names)

    plt.title(experiment_name)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix for " + experiment_name)
    plt.suptitle(metric_statement, y=1.05)
    plt.savefig(cm_path)
    plt.show()

    document_testing_metrics(experiment_name, test_loss, test_accuracy, test_f1_score, cm_path)
    return cm

############## TEST NEW IMAGE ################
def test_new_image(experiment_name, model, image_path):
    """
    Perform inference on a new image using a given model.

    Parameters:
        experiment_name (str): The name of the experiment or model being used.
        model (torch.nn.Module): The pre-trained model to use for inference.
        image_path (str): The path to the image file to be tested.

    Returns:
        None

    Prints the top prediction and a list of the top 5 predictions with their labels and confidence percentages.
    """
    img = Image.open(image_path).convert('RGB')
    transform_norm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # get normalized image
    img_normalized = transform_norm(img).float()
    img_normalized = img_normalized.unsqueeze_(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    img_normalized = img_normalized.to(device)

    with torch.no_grad():
        model.eval()
        output = model(img_normalized)
        labels = ["comfort", "danger", "death", "excitement", "fitness", "freedom", "power", "safety"]
        _, index = torch.max(output, 1)
        percentage = torch.nn.functional.softmax(output, dim=1)[0] * 100
        print("The top prediction with the model ", experiment_name)
        print(labels[index[0]], percentage[index[0]].item())

        _, indices = torch.sort(output, descending=True)
        label_list = [(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]
        print(label_list)
    return

