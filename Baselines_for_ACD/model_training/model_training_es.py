"""
This script contains functions for training and evaluating a multi-class classifier model.

Functions:

load_pretrained_model(model_name): Loads a pretrained model based on the specified model_name.
document_experiment(data_prep_type, experiment_name, base_model_name, epochs, optimizer, test_loss, test_accuracy, test_f1_score, plot_file, augmented_data): Documents the results of an experiment by writing the experiment details to a CSV file.
multi_class_classifier(data_prep_type, train_loader, val_loader, test_loader, base_model_name, model, experiment_name, epochs, augmented_data): Trains a multi-class classifier model and evaluates it on the test dataset.
"""

# ------------------IMPORT MODULES-----------------
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# Set the random seed for reproducibility
seed = 123
torch.manual_seed(seed)

def load_pretrained_model(model_name):
    """
    Loads a pretrained model based on the specified model_name.

    Args:
        model_name (str): Name of the model to load. Supported options are: "vgg16", "resnet50", "vit", "mlp_mixer".

    Returns:
        torch.nn.Module: Pretrained model instance.

    Raises:
        ValueError: If the specified model_name is not supported.
    """
    if model_name == "vgg16":
        # Load the VGG model
        model = torchvision.models.vgg16(pretrained=True)
        # model = torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    elif model_name == "resnet50":
        model = torchvision.models.resnet50(pretrained=True)
    elif model_name == "vit":
        model = timm.create_model('vit_base_patch16_224', pretrained=True)
    elif model_name == "mlp_mixer":
        model = timm.create_model('mlp_mixer_b16_224', pretrained=True)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")
    return model

def document_experiment(data_prep_type, experiment_name, base_model_name, epochs, optimizer, test_loss, test_accuracy, test_f1_score, plot_file, augmented_data):
    """
    Documents the results of an experiment by writing the experiment details to a CSV file.

    Args:
        data_prep_type (str): Type of data preparation performed for the experiment.
        experiment_name (str): Name of the experiment.
        base_model_name (str): Name of the base model used in the experiment.
        epochs (int): Number of epochs the model was trained for.
        optimizer (str): Name of the optimizer used for training.
        test_loss (float): Loss achieved on the test dataset.
        test_accuracy (float): Accuracy achieved on the test dataset.
        test_f1_score (float): F1 score achieved on the test dataset.
        plot_file (str): Path to the metrics image file.
        augmented_data (bool): Indicates whether augmented data was used in the experiment.
    """
    # Generate the filename with today's date
    filename = f"../Local_logs/experiment_documentation/experiments_{datetime.date.today()}.csv"
    # Check if the file exists
    exists = os.path.exists(filename)
    # If the file does not exist, create it and write the header row
    if not exists:
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['experiment_name', 'data_prep_type', 'augmented_data', 'base_model', 'epochs', 'optimizer',
                             'test_loss', 'test_accuracy', 'test_f1_score', 'metrics_img_path'])
    # Write the data for the current experiment
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([experiment_name, data_prep_type, augmented_data, base_model_name, epochs, optimizer, test_loss, test_accuracy, test_f1_score, plot_file])
    return

def multi_class_classifier(data_prep_type, train_loader, val_loader, test_loader, base_model_name, model, experiment_name, epochs, augmented_data):
    """
    Trains a multi-class classifier model and evaluates it on the test dataset.

    Args:
        data_prep_type (str): Type of data preparation performed for the experiment.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        base_model_name (str): Name of the base model used in the experiment.
        model (torch.nn.Module): Pretrained model instance.
        experiment_name (str): Name of the experiment.
        epochs (int): Number of epochs to train the model for.
        augmented_data (bool): Indicates whether augmented data was used in the experiment.

    Returns:
        tuple: A tuple containing the trained multi-class classifier model and the path to the saved model.
    """

    # Initialize a summary writer to log the parameters and metrics
    writer = SummaryWriter('Local_logs/metrics_plots/summary_logs/' + experiment_name)

    # Freeze the model weights
    for param in model.parameters():
        param.requires_grad = False

    # Replace the last layer to fit the number of target classes
    num_classes = 8
    try:
        model.classifier[-1] = torch.nn.Linear(in_features=4096, out_features=num_classes)
    except:
        if hasattr(model, "fc"):
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        else:
            model.head = torch.nn.Linear(model.head.in_features, num_classes)

    # Train the model
    criterion = torch.nn.CrossEntropyLoss()
    try:
        optimizer = torch.optim.Adam(model.classifier.parameters())
    except:
        if hasattr(model, "fc"):
            optimizer = torch.optim.Adam(model.fc.parameters())
        else:
            optimizer = torch.optim.Adam(model.parameters())
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    train_f1_scores = []
    val_f1_scores = []
    model.cuda()

    # for early stopping
    best_loss = float('inf')
    patience = 1000
    counter = 0
    for epoch in range(epochs):
        # for early stopping
        running_loss = 0.0
        print(f"Epoch {epoch + 1}/{epochs}")
        model.train()
        train_loss = 0
        train_accuracy = 0
        train_f1_score = 0
        print("length of train dataset being used: ", len(train_loader))
        for i, (inputs, labels) in enumerate(tqdm(train_loader)):
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            # for early stopping
            running_loss += loss.item() * inputs.size(0)
            train_loss += loss.item()
            train_pred = torch.argmax(outputs, dim=1).unsqueeze(1)
            train_accuracy += accuracy_score(labels.cpu().numpy(), train_pred.cpu().numpy().flatten())
            # train_f1_score += f1_score(labels.cpu().numpy(), train_pred.cpu().numpy().argmax(axis=1), average='micro')
            train_f1_score += f1_score(labels.cpu().numpy(), train_pred.cpu().numpy().flatten(), average="micro")
            if i % 100 == 0:
                print(f"Iteration {i}, Train Loss: {loss.item():.4f}, Train accuracy: {train_accuracy / (i + 1):.4f}, Train F1 Score: {train_f1_score / (i + 1):.4f}")

        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_accuracy / len(train_loader))
        train_f1_scores.append(train_f1_score / len(train_loader))

        with torch.no_grad():
            model.eval()
            val_loss = 0
            val_accuracy = 0
            val_f1_score = 0
            for i, (inputs, labels) in enumerate(tqdm(val_loader)):
                inputs = inputs.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels.to(device))

                val_loss += loss.item()
                val_pred = torch.argmax(outputs, dim=1)
                val_accuracy += accuracy_score(labels.cpu().numpy(), val_pred.cpu().numpy())
                val_f1_score += f1_score(labels.cpu().numpy(), val_pred.cpu().numpy(), average='micro')

                if i % 100 == 0:
                    print(f"Iteration {i}, Validation Loss: {loss.item():.4f}, Validation accuracy: {val_accuracy / (i + 1):.4f}, Validation F1 Score: {val_f1_score / (i + 1):.4f}")

        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_accuracy / len(val_loader))
        val_f1_scores.append(val_f1_score / len(val_loader))

        # Log the parameters and metrics at each epoch
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_accuracy, epoch)
        writer.add_scalar('F1Score/Train', train_f1_score, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)
        writer.add_scalar('F1Score/Validation', val_f1_score, epoch)

        # Log the model's parameters
        for name, param in model.named_parameters():
            writer.add_histogram(name, param, epoch)

        # for early stopping
        epoch_loss = running_loss / len(train_loader)
        # check if the current epoch's loss is the best so far
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            counter = 0
            path_to_save = str('../Local_logs/saved_models/' + experiment_name + 'best_model.pth')
            torch.save(model.state_dict(), path_to_save)
        else:
            counter += 1
            if counter >= patience:
                print('Early stopping at epoch', epoch)
                break

    # Plot the training history
    plt.figure(figsize=(12, 4))
    plt.suptitle(experiment_name)
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.title('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies, label='train')
    plt.plot(val_accuracies, label='val')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(train_f1_scores, label='train')
    plt.plot(val_f1_scores, label='val')
    plt.title('F1 Score')
    plt.legend()

    if not os.path.exists('../Local_logs/metrics_plots/train_history_plots/'):
        os.mkdir('../Local_logs/metrics_plots/train_history_plots/')
    plot_file = '../Local_logs/metrics_plots/train_history_plots/' + experiment_name + '.png'
    plt.savefig(plot_file)

    # Evaluate the model on the test dataset
    test_loss = 0
    test_accuracy = 0
    test_f1_score = 0

    # test evaluation
    with torch.no_grad():
        model.eval()
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.to(device))

            test_loss += loss.item()
            test_pred = torch.argmax(outputs, dim=1)
            test_accuracy += accuracy_score(labels.cpu().numpy(), test_pred.cpu().numpy())
            test_f1_score += f1_score(labels.cpu().numpy(), test_pred.cpu().numpy(),
                                                                average='weighted')

    test_loss /= len(test_loader)
    test_accuracy /= len(test_loader)
    test_f1_score /= len(test_loader)

    # Log the test loss and accuracy
    writer.add_scalar('Loss/Test', test_loss, epoch)
    writer.add_scalar('Accuracy/Test', test_accuracy, epoch)
    writer.add_scalar('WeightedF1Score/Test', test_f1_score, epoch)


    print('Test Loss: {:.4f} Test Accuracy: {:.4f} Test F1 Score: {:.4f}'.format(
        test_loss, test_accuracy, test_f1_score))

    multi_class_model = model

    # Save
    path_to_save = '../Local_logs/saved_models/' + experiment_name + 'best_model.pth'
    if not os.path.exists(os.path.dirname(path_to_save)):
        os.makedirs(os.path.dirname(path_to_save))
        torch.save(multi_class_model, path_to_save)
    writer.close()
    document_experiment(data_prep_type, experiment_name, base_model_name, epochs, optimizer, test_loss, test_accuracy, test_f1_score, plot_file, augmented_data )
    return multi_class_model, path_to_save
