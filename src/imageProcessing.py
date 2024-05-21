from PIL import Image
import os

image_path = ""
image_path_up = os.path.normpath(image_path)

filenamelst = []

for folder_path in os.listdir(image_path_up):

    for filename in os.listdir(os.path.join(image_path_up,folder_path)):

        try:
            image = Image.open(os.path.join(image_path_up,folder_path, filename))
        except Exception as e:
            print(f"Error in file {filename}: {e}")
            os.remove(os.path.join(image_path_up,folder_path, filename))
            filenamelst.append(folder_path+"\\"+filename)


print("Total files removed : ",len(filenamelst))
print(filenamelst)