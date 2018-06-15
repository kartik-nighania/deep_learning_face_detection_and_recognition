# Usage 
'''
python working_face_detection_pose_alignment_MTCNN.py --image_size 300 --margin 32 \
--random_order --gpu_memory_fraction 0.3 --detect_multiple_faces true
'''

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import tensorflow as tf
from scipy import misc
from time import sleep
import numpy as np
import facenet
import argparse
import time
import cv2
import sys
import os

# user made files
from face_aligner import FaceAligner
import align.detect_face

def main(args):
    
    print('Creating networks and loading parameters')
    
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
    
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    # Create an object of face aligner module
    affine = FaceAligner(desiredLeftEye=(0.39, 0.39), desiredFaceWidth=256, desiredFaceHeight=256)

    print("[INFO] camera sensor warming up...")
    vs = cv2.VideoCapture(0)
    vs.set(3, 1280)
    vs.set(4, 720)
    time.sleep(2.0)

    while True:
        ret, img = vs.read()

        # we get the bounding boxes as well as the points for the face
        bb, points = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        #print("here they are \n")
        #print(points)

        # See if face is detected
        if bb.shape[0] > 0:

            # Draw rectangles on the faces and circle on the the landmarks
          for i in range(bb.shape[0]):
             cv2.rectangle(img, (int(bb[i][0]),int(bb[i][1])), (int(bb[i][2]),int(bb[i][3])), (0, 255, 0), 2)

          # loop over the (x, y)-coordinates for the facial landmarks
          # and draw each of them
          for col in range(points.shape[1]):
             for i in range(5):
                cv2.circle(img, (int(points[i][col]), int(points[i+5][col])), 1, (255, 0, 0), -1)

          # ALIGNMENT - use the bounding boxes and facial landmarks to align images
          aligned_image = affine.align(img, points)

          # Show the image only if alignment is there
          cv2.imshow("Alignment", aligned_image)


        cv2.imshow("Output", img)

        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
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
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
