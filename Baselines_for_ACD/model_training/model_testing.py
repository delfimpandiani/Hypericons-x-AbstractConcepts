import torch
import csv
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, accuracy_score

device = "cuda"
############## LOAD TRAINED MODEL ################
def load_trained_model(experiment_name):
    model_PATH = "models/" + experiment_name
    model = torch.load(model_PATH)
    return model

def document_testing_metrics(experiment_name, test_loss, test_accuracy, test_f1_score, cm_path):
    # Check if the file exists
    try:
        with open('testing_metrics.csv', 'r') as f:
            exists = True
    except FileNotFoundError:
        exists = False

    # If the file does not exist, create it and write the header row
    if not exists:
        with open('testing_metrics.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(
                ["experiment_name", "test_loss", "test_accuracy", "test_f1_score", "cm_path"])

    # Write the data for the current experiment
    with open('testing_metrics.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([experiment_name, test_loss, test_accuracy, test_f1_score, cm_path])
    return

def get_confusion_matrix(experiment_name, model, test_loader, class_names):
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
    cm_path = 'confusion_matrices/' + experiment_name
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
def test_new_image(experiment_name, trained_model, image_path):
    img = Image.open(image_path).convert('RGB')
    transform_norm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # get normalized image
    img_normalized = transform_norm(img).float()
    img_normalized = img_normalized.unsqueeze_(0)
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

