from PIL import Image

def resize_images(images, size):
    resized_images = []
    for img in images:
        resized_img = img.resize(size, Image.ANTIALIAS)
        resized_images.append(resized_img)
    return resized_images

# Function to combine images horizontally
def combine_images_horizontally(images):
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)

    new_image = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for img in images:
        new_image.paste(img, (x_offset, 0))
        x_offset += img.width

    return new_image

# Function to combine images vertically
def combine_images_vertically(images):
    widths, heights = zip(*(i.size for i in images))
    max_width = max(widths)
    total_height = sum(heights)

    new_image = Image.new('RGB', (max_width, total_height))

    y_offset = 0
    for img in images:
        new_image.paste(img, (0, y_offset))
        y_offset += img.height

    return new_image

# Paths to your images
image_paths = [
    r"./report_collage/grape-black-rot-2.jpg",  # Replace with the actual path to your image A
    r"./report_collage/gumosis41_.jpg",  # Replace with the actual path to your image B
    r"./report_collage/train-cbb-19.jpg",  # Replace with the actual path to your image C
    r"./report_collage/grape-black-rot-2.JPG_trans.jpg",  # Replace with the actual path to your transformed image A
    r"./report_collage/gumosis41__trans.jpg",  # Replace with the actual path to your transformed image B
    r"./report_collage/train-cbb-19_trans.jpg"   # Replace with the actual path to your transformed image C
]

# Load images
images = [Image.open(img_path) for img_path in image_paths]

# Resize images to have the same dimensions
max_size = max(image.size for image in images)
images = resize_images(images, max_size)


# Load images
# images = [Image.open(img_path) for img_path in image_paths]

# Combine images horizontally
first_row_image = combine_images_horizontally(images[:3])
second_row_image = combine_images_horizontally(images[3:])

# Combine first and second row vertically
final_image = combine_images_vertically([first_row_image, second_row_image])

# Save the final image
final_image.save("./report_collage/combined_image.jpg")  # Replace with the desired path to save the combined image
