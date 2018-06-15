# Usage 
'''
python working_face_detection_pose_alignment_MTCNN.py --image_size 300 --margin 32 \
--random_order --gpu_memory_fraction 0.9 --detect_multiple_faces true \
--model 20180402-114759/20180402-114759.pb
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

    # Building seperate graphs for both the networks
    g1 = tf.Graph()
    g2 = tf.Graph()
    #images_placeholder = tf.placeholder(tf.int32)
    #embeddings = tf.Variable()
    #phase_train_placeholder = tf.placeholder(tf.bool)

    
    with g1.as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with tf.Session() as sess:
            facenet.load_model(args.model)
    #with tf.Graph().as_default():
        #with tf.Session() as sess:

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

    # Load the model for FaceNet image recognition and get the tensors
    

    print("[INFO] camera sensor warming up...")
    vs = cv2.VideoCapture(0)
    vs.set(3, 640)
    vs.set(4, 480)
    time.sleep(2.0)

    while True:
        ret, img = vs.read()

        # we get the bounding boxes as well as the points for the face
        g2.as_default()
        with tf.Session(graph=g2) as sess:
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

          # Prewhiten the image for facenet architecture to give better results
          mean = np.mean(aligned_image)
          std  = np.std(aligned_image)
          std_adj = np.maximum(std, 1.0/np.sqrt(aligned_image.size))
          facenet_image = np.multiply(np.subtract(aligned_image, mean), 1/std_adj)
          img_list = []
          img_list.append(facenet_image)
          img_list.append(facenet_image)
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
              print(embedding)
              print("\n")

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
    parser.add_argument('--model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
