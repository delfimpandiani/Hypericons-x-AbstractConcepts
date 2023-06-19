import torch
import torchvision
from torchvision import models, transforms
from PIL import Image as PilImage
from omnixai.data.image import Image
from omnixai.explainers.vision.specific.gradcam.pytorch.gradcam import GradCAM

# Load the trained model
model_path = "../finetuned_VGG_1000_epochs.pth"
model = torchvision.models.vgg16()
num_classes = 8
model.classifier[-1] = torch.nn.Linear(in_features=4096, out_features=num_classes)
model.load_state_dict(torch.load(model_path))

# Define preprocessing of unseen images
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
preprocess = lambda ims: torch.stack([transform(im.to_pil()) for im in ims])

# Load unseen image and preprocess to tensor
img = Image(PilImage.open('test_imgs/triumph.png').convert('RGB'))
img_tensor = preprocess([img])

# Define the GradCAM explainer
explainer = GradCAM(
    model=model,
    target_layer=model.features[-1],
    preprocess_function=preprocess
)

# Predict the label for the image
class_names = ["comfort", "danger", "death", "excitement", "fitness", "freedom", "power", "safety"]
with torch.no_grad():
    outputs = model(img_tensor)
    _, predicted = torch.max(outputs, 1)
print("Predicted label: ", class_names[predicted.item()])

## Grad-CAM for predicted label
# label_of_interest = predicted

## Grad-CAM for another label of interest
label_of_interest = 6

# Execute and plot the Grad-CAM for the label of interest
explanations = explainer.explain(img, label_of_interest)
explanations.ipython_plot(index=0, class_names=class_names)
print("Grad-CAM for label: ", class_names[label_of_interest])
