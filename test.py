import os
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image

# Define the transformations
def repeat_to_rgb(x):
    return x.repeat(3, 1, 1)  # Repeat the single grayscale channel to 3 channels (RGB)

normalize = transforms.Normalize(mean=[0.485], std=[0.229])  # For grayscale normalization

transform_pipeline = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.Resize(256),                      # Resize image
    transforms.CenterCrop(224),                  # Center crop to 224x224
    transforms.ToTensor(),                       # Convert to PyTorch tensor
    transforms.Lambda(repeat_to_rgb),            # Repeat grayscale channel to 3 channels
    normalize                                    # Normalize with ImageNet grayscale stats
])

# Load an example image
image_path = os.path.join("Data_prepared", "1gliedrig_100_files_12_views", "train","1gliedrig","1gliedrig_(10)", "0.png") # Replace with the path to your image
image = Image.open(image_path)

# Apply transformations step-by-step
transformed_images = []
titles = []

# Original image
transformed_images.append(image)
titles.append("Original Image")

# Grayscale conversion
grayscale_image = transforms.Grayscale(num_output_channels=1)(image)
transformed_images.append(grayscale_image)
titles.append("Grayscale")

# Resize
resized_image = transforms.Resize(256)(grayscale_image)
transformed_images.append(resized_image)
titles.append("Resized (256)")

# Center Crop
cropped_image = transforms.CenterCrop(224)(resized_image)
transformed_images.append(cropped_image)
titles.append("Center Crop (224x224)")

# Tensor conversion
tensor_image = transforms.ToTensor()(cropped_image)
transformed_images.append(tensor_image.permute(1, 2, 0).numpy())  # Convert for display
titles.append("Tensor (Unnormalized)")

# Repeat channel
repeated_image = repeat_to_rgb(tensor_image)
transformed_images.append(repeated_image.permute(1, 2, 0).numpy())  # Convert for display
titles.append("Repeated to 3 Channels")

# Normalize
normalized_image = normalize(repeated_image)
# Denormalize for visualization (optional, so it’s easier to interpret)
denormalized_image = normalized_image * torch.tensor([0.229, 0.229, 0.229]).view(3, 1, 1) + \
                     torch.tensor([0.485, 0.485, 0.485]).view(3, 1, 1)
transformed_images.append(denormalized_image.permute(1, 2, 0).numpy())  # Convert for display
titles.append("Normalized (Denormalized for View)")

# Apply the full transformation pipeline
fully_transformed_image = transform_pipeline(image)
# Denormalize full image for visualization
fully_transformed_image = fully_transformed_image * torch.tensor([0.229, 0.229, 0.229]).view(3, 1, 1) + \
                          torch.tensor([0.485, 0.485, 0.485]).view(3, 1, 1)
transformed_images.append(fully_transformed_image.permute(1, 2, 0).numpy())  # Convert for display
titles.append("Fully Transformed")

# Plot all transformations
plt.figure(figsize=(15, 10))
for i, img in enumerate(transformed_images):
    plt.subplot(2, 4, i + 1)
    if isinstance(img, torch.Tensor):  # If tensor, convert to numpy for visualization
        img = img.permute(1, 2, 0).numpy()
    plt.imshow(img, cmap="gray" if i == 1 else None)  # Use grayscale colormap for grayscale images
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()