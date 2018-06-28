import cv2
import os

image_folder = input("\nWrite the folder path inside which images are kept: ").rstrip().lstrip()

out_folder = input("\nWrite the output folder path inside which images to be saved or press ENTER to create in this directory only: ").rstrip().lstrip()
if out_folder == "":
	out_folder = "resized_images"
else:
	out_folder += "/resized_images"

# create folder
os.mkdir(out_folder)

res = (30,30)
ask = input("\nMention the resolution of images in widthxheight format or press ENTER for Default 30x30:")
if ask != "":
    res = (int(ask.split("x")[0]), int(ask.split("x")[1]))

for img_path in os.listdir(image_folder):
    image = cv2.imread(image_folder + "/" + img_path)
    image = cv2.resize(image, res)

    # Final path
    out_path = out_folder + "/" + img_path
    #.split(".")[0] + ".jpg"
    cv2.imwrite(out_path, image)

print("\nImages succesfully saved in "+ out_folder + "\n")