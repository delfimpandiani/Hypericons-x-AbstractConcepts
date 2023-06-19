"""
Dataset Prep and Splitting Script

This script is used to split a given dataset into three parts: train, validation, and test datasets. The splitting process can be customized based on the data preparation type and whether augmented data is used.

Usage:
    - Specify the data_prep_type parameter to control the dataset splitting strategy.
    - Set augmented_data parameter to True if augmented training data is desired.

"""

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
    """
    Prepare the data for training and evaluation.

    Args:
        dataset_path (str): : Local path to the directory that contains the images, separated in folders with labels as names
        data_prep_type (str): Type of data preparation. Possible values: "ub" (unbalanced classes), "overall_b" (overall balanced), "partition_b" (partition balanced).
        augmented_data (bool): Flag to indicate whether augmented training data should be used.

    Returns:
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        test_loader (DataLoader): DataLoader for the test dataset.
        train_dataset (Subset): Subset of the training dataset.
        val_dataset (Subset): Subset of the validation dataset.
        test_dataset (Subset): Subset of the test dataset.
    """

    # Locate the dataset
    dataset = torchvision.datasets.ImageFolder(root=dataset_path)

    # Create a dict with classes as keys and instances as values
    class_indices = {}
    for idx, (_, label) in enumerate(dataset):
        if label not in class_indices:
            class_indices[label] = [idx]
        else:
            class_indices[label].append(idx)
    print(class_indices.keys())

    # Define the split ratios
    train_ratio = 0.8
    val_ratio = 0.1


    ## DATA BALANCING: UNBALANCED, OVERALL BALANCED, OR PARTITION BALANCED
    if data_prep_type == "ub": # unbalanced classes: if we want to keep all dataset images per class
        selected_indices = []
        for label, indices in class_indices.items():
            unbalanced_indices_for_label = indices[:]
            print("When dataset is unbalanced, ", label, " has this number of instances: ", len(unbalanced_indices_for_label))
        print("Total number of instances: ", len(dataset))

        # Calculate the length of the datasets
        dataset_length = len(dataset)
        train_length = int(train_ratio * dataset_length)
        val_length = int(val_ratio * dataset_length)
        test_length = dataset_length - train_length - val_length

        # Split the dataset
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_length, val_length, test_length])
    else: # if we want the dataset to be balanced (max 1000 images per class); below it is possible to choose overall balanced or partition balanced
        selected_indices = []
        for label, indices in class_indices.items():
            if len(indices) < 1000:
                balanced_indices_for_label = indices
            else:
                balanced_indices_for_label = random.sample(indices, 1000)
            selected_indices.extend(balanced_indices_for_label)
            print("When dataset is overall balanced, ", label, " has this number of indices ", len(balanced_indices_for_label))

        # Calculate the length of the datasets
        selected_indices_length = len(selected_indices)
        train_length = int(train_ratio * selected_indices_length)
        val_length = int(val_ratio * selected_indices_length)
        test_length = selected_indices_length - train_length - val_length

        if data_prep_type == "overall_b": # overall balanced: max 1000 images per class is the only important thing
            # Split the dataset
            random.shuffle(selected_indices)
            train_indices = selected_indices[:train_length]
            val_indices = selected_indices[train_length:train_length + val_length]
            test_indices = selected_indices[train_length + val_length:]

            # Create the train, validation, and test datasets
            train_dataset = torch.utils.data.Subset(dataset, train_indices)
            val_dataset = torch.utils.data.Subset(dataset, val_indices)
            test_dataset = torch.utils.data.Subset(dataset, test_indices)

            print("Overall balanced train dataset length: ", len(train_dataset))
            print("Overall balanced val dataset length: ", len(val_dataset))
            print("Overall balanced test dataset length: ", len(test_dataset))

        elif data_prep_type == "partition_b": # partition balanced: max 1000 images per class AND make sure that the train, val and test datasets are each internally class balanced
            # Split the dataset
            train_indices = []
            val_indices = []
            test_indices = []
            for label, indices in class_indices.items():
                random.shuffle(indices)
                class_train_indices = indices[:train_length // len(class_indices.keys())]
                class_val_indices = indices[train_length // len(class_indices.keys()):train_length // len(
                    class_indices.keys()) + val_length // len(class_indices.keys())]
                class_test_indices = indices[train_length // len(class_indices.keys()) + val_length // len(
                    class_indices.keys()):train_length // len(class_indices.keys()) + val_length // len(
                    class_indices.keys()) + test_length // len(class_indices.keys())]

                train_indices.extend(class_train_indices)
                val_indices.extend(class_val_indices)
                test_indices.extend(class_test_indices)

            # Create the train, validation, and test datasets
            train_dataset = torch.utils.data.Subset(dataset, train_indices)
            val_dataset = torch.utils.data.Subset(dataset, val_indices)
            test_dataset = torch.utils.data.Subset(dataset, test_indices)

            print("Partition balanced train dataset length: ", len(train_dataset))
            print("Partition balanced val dataset length: ", len(val_dataset))
            print("Partition balanced test dataset length: ", len(test_dataset))

    ## DATA AUGMENTATION
    # Define data transforms for augmenting the training data
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)], p=0.3),
        transforms.RandomApply([transforms.RandomRotation(15)], p=0.3),
        transforms.RandomApply([transforms.RandomCrop(224, padding=20)], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Define data transforms for the validation and test sets
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    if augmented_data == True:
        # Apply the appropriate transforms to the datasets, including the TRAINING transform
        print("I am working with augmented training data")
        train_dataset.dataset.transform = train_transform
        val_dataset.dataset.transform = val_transform
        test_dataset.dataset.transform = val_transform
    else:
        print("I am NOT working with augmented training data")
        # Apply the appropriate transforms (basic one for validation) to the datasets
        train_dataset.dataset.transform = val_transform
        val_dataset.dataset.transform = val_transform
        test_dataset.dataset.transform = val_transform

    # Create the dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset
