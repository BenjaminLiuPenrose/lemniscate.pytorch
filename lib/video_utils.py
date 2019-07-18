import numpy as np
import torch
import h5py
from pdb import set_trace as st
import torch.nn as nn
import argparse
from torch.autograd import Variable
import pickle
import os
import cv2


def process_vidseq(movie_path, interval=1):
    """ process one video
    """
    cap = cv2.VideoCapture(movie_path)
    frame_all = []
    count = 0
    while(cap.isOpened()):
        # Capture frame-by-frame
        success, frame = cap.read()
        if success == True:
            frame_all.append(frame)
        else:
            break
        count += 1
    return np.array(frame_all[::interval])

def frameArr_to_frameJpg(frame_array, jpg_path_parent):
    """ write from np array to jpg
    """
    length = len(frame_array)
    for i in range(length):
        jpg_path = os.path.join(jpg_path_parent, "frame%d.jpg" % i)
        frame = frame_array[i]
        cv2.imwrite(jpg_path, frame)
    return True


def test():
    movie_path_datahome = "../data/UCF-101/"
    jpg_path_datahome = "../data/UCF-101-Frame/"
    suffix = ".avi"
    instance_idx = 0

    movieCategoryName_ls = [folder[0].split("/")[-1] for folder in os.walk(movie_path_datahome)]
    for movieCategoryName in movieCategoryName_ls:
        # movieCategoryName = "ApplyEyeMakeup"
        movie_path_parent = os.path.join(movie_path_datahome, movieCategoryName)
        filenames = os.listdir(movie_path_parent)
        moviename_ls = [ filename for filename in filenames if filename.endswith( suffix ) ]
        for moviename in moviename_ls:
            moviename_rmsuffix = moviename[:-len(suffix)]
            instance_name = "Instance{}_{}".format(instance_idx, moviename_rmsuffix)
            movie_path = os.path.join(movie_path_parent, moviename)
            jpg_path_parent = os.path.join(jpg_path_datahome, instance_name)
            if not os.path.exists(jpg_path_parent):
                os.makedirs(jpg_path_parent)
            frame_array = process_vidseq(movie_path)
            success = frameArr_to_frameJpg(frame_array, jpg_path_parent)
            print("Finish process {}".format(instance_name))
            instance_idx += 1

if __name__ == "__main__":
    test()
