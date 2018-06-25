# This script takes a video input and provides option to see and save any number of frame as an image.

import numpy as np
import os
import cv2

def main():
     path = input("\nEnter the filename of the video: ").lstrip().rstrip().replace(" ","_")
     # Create video object
     cap = cv2.VideoCapture(path)

     # remove the .mp4 etc extension from the path name
     path = path.split(".")[0]

     print("\nINSTRUCTIONS\n\nPress d to move forward\nPress a to move backward.\nPress s to save the image\nPress q on the image window to close the video.\n")

     folder_name = input("\nEnter the folder name in which labeled folder_name_xx.jpg images will be saved by the user: ")
     if os.path.isdir(folder_name):
        print("folder already exists. Using it \n")
     else:
        os.makedirs(folder_name)

     frameNo= 0
     prevNo = -1
     image_no = 1
     image_path = ""
     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
     print("\nTotal number of frames: " + str(total_frames))

     # Close if video ends or user pressed q keyword
     while ((cv2.waitKey(10) & 0xFF) != ord("q")) and (frameNo < total_frames) and frameNo >= 0 :
        
         # Set the frame number to access
         cap.set(1,frameNo)
         # Read the frame
         _, frame = cap.read()
         # show frame on window
         cv2.imshow(path, frame)

         # See for a change in frame number to print
         if frameNo != prevNo:
            print("\nFrame number is: "+ str(frameNo))

         key = (cv2.waitKey(1000) & 0xFF)
         
         prevNo = frameNo

         if key == ord("d"):
            frameNo += 1
         if key == ord("a"):
            frameNo -= 1
         # Save the image
         if key == ord("s"):
            image_path = path + "/" + path + "_" + str(image_no).zfill(4) + ".jpg"
            print("\nhere it is: ", image_path)
            cv2.imwrite(image_path, frame)
            image_no += 1

     # Clean and exit
     cap.release()
     cv2.destroyAllWindows()

if __name__ == '__main__':
  main()