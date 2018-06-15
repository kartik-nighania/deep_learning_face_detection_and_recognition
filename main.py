from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import tensorflow as tf
from scipy import misc
from time import sleep
import numpy as np
import argparse
import keyboard
import facenet
import time
import cv2
import sys
import os

# user made files
from face_aligner import FaceAligner
import align.detect_face

def main():

  print("\n*********************************************************************************************** \n")
  print("              Welcome to the Face detection and recognition program. \n")
  print("\n*********************************************************************************************** \n")
  print("GUIDELINES TO USE THIS SOFTWARE: \n\nThis code gives the user to:\n\n1) CREATE DATASET using MTCNN face detection and alignment. or\n2) TRAIN FaceNet for face recognition. or \n3) Do both.\n\n The user will multiple times get option to choose webcam (default option) or video file to do face detection and will be asked for output folder, username on folder and image files etc also (default options exists for that too)\n\n **************   IMPORTANT   *************\n1) Whenever webcam or video starts press 's' keyword to start face detection in video or webcam frames and save the faces in the folder for a single user. This dataset creation will stop the moment you release the 's' key. This can be done multiple times.\n\n2) Press 'q' to close it when you are done with one person, and want to detect face for another person. \n\n3) Make sure you press the keywords on the image window and not the terminal window.\n")
  mode = input("Press T to train the facenet for recognition OR \nPress D to first create dataset and then 'maybe' train later: ")

  # Some variables that will be used through out the code
  path = ""
  res = ()
  personNo = 1
  folder_name = ""


  # This means user went for Creating of dataset
  if mode == 'D':
    path = input("Enter the output folder location or simply press ENTER create a dataset folder in this directory only: ")

    if os.path.isdir(path):
     
     # User given path is present.
      path += '/output'
      if os.path.isdir(path):
        print("Directory already exists. Using it \n")
      else:
        if not os.makedirs(path):
          print("Directory successfully made in: " + path + "\n")

     # either user pressed ENTER or gave wrong location.
    else:
       if path == "":
         print("Making an output folder in this directory only. \n")

       else:
           print("No such directory exists. Making an output folder in this current code directory only. \n")

       path = 'output'
       if os.path.isdir(path):
       	 print("Directory already exists. Using it \n")
       else:
          if os.makedirs(path):
       	    print("error in making directory. \n")
            sys.exit()
          else:
             print("Directory successfully made: " + path + "\n")

    # Ask for webcam resolution
    res = tuple(map(int, input("Enter your webcam SUPPORTED resolution for face detection. For eg. 640x480 OR press ENTER for default 640x480: ").split("x")))
    if res == "":
      res = (640, 480)

    # Start MTCNN face detection and pose estimation module.
    
    # Take gpu fraction values
    gpu_fraction = input("\nEnter the gpu memory fraction u want to allocate out of 1 or press ENTER for default 0.8: ")
    if gpu_fraction == "":
      gpu_fraction = 0.8
    else:
    	gpu_fraction = round(float(gpu_fraction), 1)

    # Some more MTCNN parameter
    minsize = 20 # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # Three steps's threshold
    factor = 0.709 # scale factor
    
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

    # Create an object of face aligner module
    face_size = tuple(map(int, input("Enter desired face width and height in widthxheight format OR press ENTER for default 160x160 pixel: ").split("x")))
    if face_size == "":
      face_size = (160, 160)
    affine = FaceAligner(desiredLeftEye=(0.33, 0.33), desiredFaceWidth=face_size[0], desiredFaceHeight=face_size[1])



  # This means user went for the train part
  elif mode == 'T':
     train()

  else:
  	print("No correct keyword entered. Exiting")
  	sys.exit()

 # Create dataset was choosen before and so working with taking dataset.
  while True:

    ask = input("\n Enter the user name for CREATING FOLDER with given username and image naming inside with username_xx.png numbered format or press ENTER to use default person_xx naming format: ")
    # removing all spaces with underscore
    ask = ask.replace(" ", "_")    

    if ask=="":
   	 folder_name = 'person_' + str(personNo)
    else:
   	  folder_name = ask

    # Creating new user specific variables   	
    personNo += 1
    users_folder = path + "/" + folder_name
    image_no = 0

    # Create folder with the given location and the given username.
    if os.path.isdir(users_folder):
       	 print("Directory already exists. Using it \n")
    else:
      if os.makedirs(path):
       	print("error in making directory. \n")
        sys.exit()
      else:
        print("Directory successfully made: " + users_folder + "\n")

    # Start webcam or videofile according to user.
    data_type = input("Press ENTER for detecting " + folder_name + " with webcam or write video path to open and create dataset of " + folder_name + " : ")

    # default webcam which uses infinite loop and video variable to find total frames
    loop_type = False
    total_frames = 0
    
    if data_type == "":
       data_type = 0
       loop_type = True

    # Initialize webcam or video
    device = cv2.VideoCapture(data_type)

    # If webcam set resolution
    if data_type == 0:
      device.set(3, res[0])
      device.set(4, res[1])
    else:
       # Finding total number of frames of video.
       total_frames = int(device.get(cv2.CAP_PROP_FRAME_COUNT))
      
      # Start web cam and creating dataset by user.
      while loop_type or (total_frames > 0):
      	 total_frames -= 1

         ret, image = device.read()

      	 # Run MTCNN and do face detection until 's' keyword is pressed
      	 if (cv2.waitKey(1) && 0xFF) == ord("s"):

           # DETECT FACES. We get the bounding boxes as well as the points for the face
           bb, points = align.detect_face.detect_face(image, minsize, pnet, rnet, onet, threshold, factor)
           
           # See if face is detected
           if bb.shape[0] > 0:
             
             # align the detected faces
             for col in range(points.shape[1]):
                aligned_image = affine.align(image, points[:,col])
                
                # Save the image
                image_name = users_folder + "/" + folder_name + "_" + str(image_no).zfill(3) + ".png"
                cv2.imwrite(image_name, aligned_image)
                image_no += 1

             # Draw the bounding boxes and pose landmarks on the image
             # Draw functions to show rectangles on the faces and circle on the the landmarks
             for i in range(bb.shape[0]):
                cv2.rectangle(image, (int(bb[i][0]),int(bb[i][1])), (int(bb[i][2]),int(bb[i][3])), (0, 255, 0), 2)

             # loop over the (x, y)-coordinates for the facial landmarks
             # and draw each of them
             for col in range(points.shape[1]):
                for i in range(5):
                   cv2.circle(image, (int(points[i][col]), int(points[i+5][col])), 1, (0, 255, 0), -1)

         # Show the output video to user
         cv2.imshow("Output", image)

         # Break this loop if 'q' keyword pressed to go to next user.
         if (cv2.waitKey(1) && 0xFF) == ord("q"):
           device.release()
           cv2.destroyAllWindows()
           break

    # Ask for more user using webcam or video else exit.
    ask = input("Press ENTER if you want to add more users or press the keyword 'q' to stop dataset creation: ")
    if ask == 'q':
      break

  # This means dataset creating is complete. ASK the user for train now or exit.
  ask = input("Press ENTER to exit or press T keyword to train the data by Facenet model on dataset: ")
  if ask = "T":
  	train()


def train():
   gpu_fraction = input("\nEnter the gpu memory fraction u want to allocate out of 1 or press ENTER for default 0.8: ")
    
   if gpu_fraction == "":
     gpu_fraction = 0.8
   else:
      gpu_fraction = round(float(gpu_fraction), 1)

   model = input("Enter the folder path where the folder 20180402-114759 model is present. Press ENTER stating that this folder is present in this code directory itself: ")
   if model == "":
      model = "20180402-114759/20180402-114759.pb"
   else:
      model += "/20180402-114759/20180402-114759.pb"

   # Load facenet face recognizer
   with tf.Graph.asDefault():
      gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
      sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
      with tf.Session() as sess:
    	   facenet.load_model(model)



if __name__ == '__main__':
  main()
