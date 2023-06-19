## OmnixAI Feature Visualization for Computer Vision Models
This Python script demonstrates the use of OmnixAI's feature visualization approach for generating visualizations of the features learned by a pre-trained VGG model in a computer vision task. The script allows you to create both regularized and unregularized feature visualizations.

### Functionality
The script performs the following steps:

- Loads a pre-trained VGG model that has been fine-tuned on a specific computer vision task.
- Defines the target layer for the activation maximization (feature visualization) as the last layer of the model's classifier.
- Defines the class names for the computer vision task.
- Creates a function, createFVs, that generates feature visualizations based on the specified regularization.
- Within the createFVs function:
- Initializes the OmnixAI FeatureVisualizer with the pre-trained model and the target layer.
- Creates a folder for each class if it doesn't already exist.
- Generates feature visualizations based on the specified regularization (regularized or unregularized).
- Saves the generated feature visualizations in the corresponding class folder.
- Calls the createFVs function with the specified regularization type.
- The script creates and saves the feature visualizations based on the chosen regularization in the appropriate class folders. 

### Regularized vs Unregularized Feature Visualizations

Feature visualizations can be generated with or without regularization. Here's an explanation of the difference between these two approaches:

#### Unregularized Feature Visualization
Unregularized feature visualizations are created without any additional regularization constraints. This means that the generated visualizations aim to maximize the activation of a specific neuron in the target layer, without imposing any explicit constraints or penalties on the image. Unregularized feature visualizations can provide insights into the visual patterns that activate specific neurons in the model.

#### Regularized Feature Visualization
Regularized feature visualizations are created by applying regularization techniques during the optimization process. In this script, several combinations of regularization parameters are tested, including different regularization methods (such as L1, L2, and Total Variation) and their corresponding weights. Regularization helps to introduce additional constraints or penalties on the image generation process, which can lead to more controlled and interpretable visualizations. Regularized feature visualizations can be useful for understanding how different regularization techniques affect the learned features and can help in identifying meaningful patterns in the images.

#### Supported Activation Maximization Parameters
The script supports the following parameters for regularized feature visualizations:

- Iterations: 300, 400, 500
- Learning Rate: 0.1, 0.01, 0.001
- Regularizer: L1, L2, TV
- Regularizer Weight: 0, −0.05, −0.5, -2.5
- Fourier Preconditioning: Yes, No
- Map Uncorrelated Colors to Normal Colors: Yes, No

#### Best Activation Maximization Parameters for Abstract Concept Hypericons
After manual inspections, the following parameters were found to yield the best results:

- Iterations: 400
- Learning Rate: 0.1
- Regularizer: TV
- Regularizer Weight: -2.5
- Fourier Preconditioning: Yes
- Map Uncorrelated Colors to Normal Colors: Yes

Please note that these parameters may vary depending on your specific use case and dataset. It is recommended to experiment with different parameter combinations to achieve the best results for your application.
### Prerequisites
To run this script, the following dependencies need to be installed:

- torch
- torchvision
- tqdm
- omnixai
- The script also assumes the availability of the pre-trained VGG model file (finetuned_VGG_1000_epochs.pth) in the parent directory.

### Usage
- Ensure that the dependencies mentioned in the prerequisites are installed.
- Place the pre-trained VGG model file (finetuned_VGG_1000_epochs.pth) in the parent directory.
- Update the class_names list with the appropriate class names for your computer vision task.
- Run the script.
- The script will generate feature visualizations for each class based on the specified regularization type.
- The feature visualizations will be saved in their respective class folders.

### Output
The script outputs the following:

- Feature visualizations for each class based on the specified regularization type (regularized or unregularized).
- The feature visualizations are saved as PNG files in their respective class folders.