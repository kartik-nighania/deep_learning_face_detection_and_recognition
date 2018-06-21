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

'''
This code takes input as video and runs:

1) face detection by MTCNN
2) Pose estimation
3) alignment, prewhitening of images
4) Saves the images in a folder specified directory
5) Also creates the video with face detection and pose landmarks
'''

# Usage 
'''
python video_input_MTCNN_detect_align_and_save_images_in_folder.py \
--video Avengers.mp4 --image_size 300 --margin 32 --random_order \
--gpu_memory_fraction 0.9 --detect_multiple_faces true \
--model 20180402-114759/20180402-114759.pb --save_video False --show_video True
'''

# user made files
from face_aligner import FaceAligner
import align.detect_face

def main(args):
    
    print('Creating networks and loading parameters')
    # Building seperate graphs for both the tf architectures
    #g1 = tf.Graph()
    g2 = tf.Graph()

    '''
    with g1.as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with tf.Session() as sess:
        	# Load the model for FaceNet image recognition
            facenet.load_model(args.model)
    '''

    with g2.as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

    
    
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    # Create an object of face aligner module
    affine = FaceAligner(desiredLeftEye=(0.33, 0.33), desiredFaceWidth=160, desiredFaceHeight=160)
    
    # Taking the video and creating an object of it.
    print("[INFO] Taking the video input.")
    vs = cv2.VideoCapture(os.path.expanduser(args.video))

    # Finding the file format, size and the fps rate
    fps = vs.get(cv2.CAP_PROP_FPS)
    video_format = int(vs.get(cv2.CAP_PROP_FOURCC))
    frame_size = (int(vs.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    total_frames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
    output_video = cv2.VideoWriter("Output_" + args.video, video_format, fps, frame_size)

    # Create the output_faces directory by user or default arguments
    path = os.path.expanduser(args.output)
    path = path+"/output_faces"

    if not os.path.isdir(path):
      os.makedirs(path)

    image_numbers = 0;

    print("Total number of frames \n" + str(total_frames) + "\n")
    #for i in range(total_frames):
    for i in range(total_frames):

        # Print the present frame / total frames to know how much we have completed
        print("\n" + str(i) + " / "+ str(total_frames) + "\n")
        
        ret, image = vs.read()

        # Run MTCNN model to detect faces
        g2.as_default()
        with tf.Session(graph=g2) as sess:
        	# we get the bounding boxes as well as the points for the face
            bb, points = align.detect_face.detect_face(image, minsize, pnet, rnet, onet, threshold, factor)

        # See if face is detected
        if bb.shape[0] > 0:

          # ALIGNMENT - use the bounding boxes and facial landmarks to align images
          # create a numpy array to feed the network
          img_list = []
          images = np.empty([bb.shape[0], image.shape[0], image.shape[1]])

          for col in range(points.shape[1]):
             aligned_image = affine.align(image, points[:,col])
             
             if args.show_video == True:
               cv2.imshow("aligned", aligned_image)

             # Prewhiten the image for facenet architecture to give better results
             #mean = np.mean(aligned_image)
             #std  = np.std(aligned_image)
             #std_adj = np.maximum(std, 1.0/np.sqrt(aligned_image.size))
             #ready_image = np.multiply(np.subtract(aligned_image, mean), 1/std_adj)
             # Save the found out images
             place = path + "/" + "output_faces_" + str(image_numbers) + ".png"
             print("saved to: " + place + "\n")
             cv2.imwrite(place, aligned_image)
             image_numbers += 1

          # if we want to show or save the video then draw the box and the points on the image
          if args.show_video == True or args.save_video == True :
            
            for i in range(bb.shape[0]):
               cv2.rectangle(image, (int(bb[i][0]),int(bb[i][1])), (int(bb[i][2]),int(bb[i][3])), (0, 255, 0), 2)

            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw each of them
            for col in range(points.shape[1]):
               for i in range(5):
                  cv2.circle(image, (int(points[i][col]), int(points[i+5][col])), 1, (255, 0, 0), -1)

        if args.save_video == True:
              output_video.write(image)

        if args.show_video == True:
              cv2.imshow("Output", image)

          # Save the final aligned face image in given format

        """   # Show the image
                #cv2.imshow(str(col), aligned_image)
                img_list.append(ready_image)
                images = np.stack(img_list)


          g1.as_default()
          with tf.Session(graph=g1) as sess:
          # Run forward pass on FaceNet to get the embeddings
              images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
              embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
              phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
              feed_dict = { images_placeholder: images, phase_train_placeholder:False }
              embedding = sess.run(embeddings, feed_dict=feed_dict)
          
              print("Here is the embedding \n")
              print(embedding.shape)
              print("\n")

        """
        

        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
        #if keyboard.is_pressed('q'):
          # do a bit of cleanup
          vs.release()
          output_video.release()
          cv2.destroyAllWindows()
          break

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--video', type=str, help='Video path')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=182)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--random_order', 
        help='Shuffles the order of images to enable alignment using multiple processes.', action='store_true')
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--detect_multiple_faces', type=bool,
                        help='Detect and align multiple faces per image.', default=True)
    parser.add_argument('--model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('--show_video', type=bool,
                        help='Show the detection and pose estimation on the real video', default=False)
    parser.add_argument('--save_video', type=bool,
                        help='Save the video with the bounding box and landmarks', default=False)
    parser.add_argument('--output', type=str,
                        help='The output path if want to specify else goes with the same directory', default=os.getcwd())
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
