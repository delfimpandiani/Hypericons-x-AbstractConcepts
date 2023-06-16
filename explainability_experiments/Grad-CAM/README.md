# OmnixAI Grad-CAM Visualization for Computer Vision Models
This Python script demonstrates the use of OmnixAI's Grad-CAM (Gradient-weighted Class Activation Mapping) approach for visualizing the regions of an image that contribute most to the predictions made by a pre-trained VGG model in a computer vision task. Grad-CAM provides insights into the model's decision-making process by highlighting the important regions.

### Functionality
The script performs the following steps:

- Loads a pre-trained VGG model that has been fine-tuned on a the abstract concept detection (ACD) task.
- Defines a preprocessing pipeline for transforming unseen images before feeding them to the model.
- Loads an unseen image for visualization and preprocesses it.
- Initializes the OmnixAI GradCAM explainer using the pre-trained model and target layer.
- Predicts the label of the unseen image using the pre-trained model.
- Generates a Grad-CAM visualization for the predicted label or a label of interest specified by the user using OmnixAI's approach.
- Displays the Grad-CAM visualization, highlighting the important regions in the image that contribute to the predicted label.

### Prerequisites
To run this script, the following dependencies need to be installed:

- torch
- torchvision
- PIL (Python Imaging Library)
- omnixai

The script also assumes the availability of the pre-trained VGG model.

### Usage
- Ensure that the dependencies mentioned in the prerequisites are installed.
- Place the pre-trained VGG model file (finetuned_VGG_1000_epochs.pth) in the baseline_detectors/models directory.
- Provide the path to the image file you want to visualize in the line img = Image(PilImage.open('test_imgs/triumph.png').convert('RGB')).
- Run the script.
- The script will output the predicted label for the image and generate a Grad-CAM visualization for the predicted label or a label of interest specified in the script using OmnixAI's Grad-CAM approach.
- The Grad-CAM visualization will be displayed in the browser, highlighting the important regions in the image.

### Output
The script outputs the following:

- The predicted label for the unseen image based on the pre-trained VGG model.
- The Grad-CAM visualization, which is a heatmap overlaid on the original image, highlighting the regions that contribute most to the label of interest.
