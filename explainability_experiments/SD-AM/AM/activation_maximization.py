import torch
import torchvision
from itertools import product
import os
from tqdm import tqdm
from omnixai.explainers.vision.specific.feature_visualization.visualizer import FeatureVisualizer

# Load the trained model
model_path = "../finetuned_VGG_1000_epochs.pth"
model = torchvision.models.vgg16()
num_classes = 8
model.classifier[-1] = torch.nn.Linear(in_features=4096, out_features=num_classes)
model.load_state_dict(torch.load(model_path))

# Define the target le=ayer for the activation maximization (feature visualization)
target_layer = model.classifier[-1]

class_names = ["comfort", "danger", "death", "excitement", "fitness", "freedom", "power", "safety"]

def createFVs(regularization):

    for index in tqdm(list(range(8))):
        optimizer = FeatureVisualizer(
            model=model,
            objectives=[{"layer": target_layer, "type": "neuron", "index": index}]
        )

        # If it doesn't exist, create a folder for each of the classes
        if not os.path.exists(class_names[index]):
            os.makedirs(class_names[index])


        if regularization == "unregularized":
            # Create the unregularized feature visualizations
            explanations = optimizer.explain(
                use_fft=False,
                normal_color=False,
                learning_rate=0.1,
                verbose=False,
                num_iterations=400,
                image_shape=(224, 224)
            )
            use_fft = False,
            normal_color = False,
            learning_rate = 0.1,
            verbose = False,
            num_iterations = 400,
            image_shape = (224, 224)
            image = explanations.explanations[0]["image"][0]
            image.save(f"{class_names[index]}/UR_{num_iterations}_{learning_rate}_fft-{use_fft}_color-{normal_color}.png")

        elif regularization == "regularized":

            # If it doesn't exist, create a folder for each of the classes
            reg_folder = str(class_names[index] + "/regularized/")
            if not os.path.exists(reg_folder):
                os.makedirs(reg_folder)

            # Determine the parameters and values to be tested for the activation maximization
            iterations = range(300, 500, 100)
            learning_rates = [0.1]
            regularizers = ["l1", "l2", "tv"]
            reg_weights = [0, -0.05, -0.5, -2.5]
            use_fft = [True]
            normal_color = [True]
            # Create regularized feature visualizations for each of the combinations
            for iteration, lr, reg, reg_weight, fft, color in product(iterations, learning_rates, regularizers, reg_weights, use_fft, normal_color):
                    explanations = optimizer.explain(
                        use_fft=fft,
                        normal_color=color,
                        regularizers=[(reg, reg_weight)],
                        learning_rate=lr,
                        verbose=False,
                        num_iterations=iteration,
                        image_shape=(224, 224)
                    )
                    image = explanations.explanations[0]["image"][0]
                    image.save(f"{class_names[index]}/regularized/{iteration}_{lr}_{reg}__{reg_weight}_fft-{fft}_color-{color}.png")


regularization = "regularized"
createFVs(regularization)