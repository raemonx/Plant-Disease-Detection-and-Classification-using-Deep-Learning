import os
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

# Define your transformations
TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.1),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
])

def apply_transforms_to_folder(input_folder, output_folder, transforms):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    transform_arr = []
    # List all files in the input folder
    files = os.listdir(input_folder)
    
    # Apply transformations to each image
    for file in files:
        # Load image
        img_path = os.path.join(input_folder, file)
        img = Image.open(img_path)
        
        # Apply transformations
        transformed_img = transforms(img)

        # transformed_img_pil = transformed_img
        
        # Convert tensor back to PIL image
        transformed_img_pil = F.to_pil_image(transformed_img)

        transform_arr.append(transformed_img_pil)

        
        
        # Save transformed image
        # output_path = os.path.join(output_folder, file.replace(".jpg","")+"_trans2.jpg")
        # transformed_img_pil.save(output_path)

        return transform_arr
        # img_grid.save(output_path)

# Function to apply transformations and save images
# def apply_transforms_to_folder(input_folder, output_folder, transforms):
#     # Create output folder if it doesn't exist
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
    
#     # List all files in the input folder
#     files = os.listdir(input_folder)

#     transformed_images = []
    
#     # Apply transformations to each image
#     for file in files:
#         # Load image
#         img_path = os.path.join(input_folder, file)
#         img = Image.open(img_path)

#         # Apply transformations
#         transformed_img = transforms(img)
        
#         # Save transformed image
#         output_path = os.path.join(output_folder, file.replace(".jpg","")+"_trans.jpg")
#         transformed_img.save(output_path)
#         transformed_images.append(img)
#         transformed_images.append(transformed_img)

#     return transformed_images



# Paths to your input and output folders
input_folder = r"./report_collage"
output_folder = r"./report_collage/"

# Apply transformations and save images
images_tr = apply_transforms_to_folder(input_folder, output_folder, TRAIN_TRANSFORMS)

# print(images_tr)

def imshow(img):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img = img.numpy()
    img = np.transpose(img, (1, 2, 0))

    plt.imshow(img)
    plt.show()
    img = img * std + mean
    plt.imshow(img)
    plt.show()

images, _ = next(iter(images_tr))
img_grid = make_grid(images)
imshow(img_grid)


# num_images = len(transformed_images)
# # num_images = 6
# num_cols = 3  # Number of columns in the collage
# # num_rows = -(-num_images // num_cols)  # Ceiling division to get the number of rows
# num_rows = 2
# # Create a new figure
# fig, axes = plt.subplots(num_rows, num_cols, figsize=(3, 3))

# # Flatten the axes array if it's not already 1-dimensional
# if num_rows == 1:
#     axes = [axes]

# # Plot each image
# for i, ax in enumerate(axes.flat):
#     if i < num_images:
#         ax.imshow(transformed_images[i])
#     ax.axis('off')

# # Remove empty subplots
# for i in range(num_images, num_rows * num_cols):
#     axes.flat[i].set_visible(False)

# plt.tight_layout()
# plt.show()
