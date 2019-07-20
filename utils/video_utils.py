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

# due to https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

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

def frameArr_to_frameJpg(frame_array, jpg_path_parent, filename_fmt, width = None, height = None):
    """ write from np array to jpg
    """
    length = len(frame_array)
    for i in range(length):
        jpg_path = os.path.join(jpg_path_parent, filename_fmt % i)
        frame = frame_array[i]
        frame = image_resize(frame, width = width, height = height)
        cv2.imwrite(jpg_path, frame)
    return True

def process_ucf101(movie_path_datahome = "../data/UCF-101/"):
    """ process ucf101 to data instance dict
    """
    # jpg_path_datahome = "../data/UCF-101-Frame/"
    suffix = ".avi"
    instance_idx = 0
    ucf101_frame_dict = {}
    cut_at_n_frame = 50
    # if not os.path.exists(jpg_path_datahome):
        # os.makedirs(jpg_path_datahome)

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
            # jpg_path_parent = os.path.join(jpg_path_datahome, instance_name)
            # if not os.path.exists(jpg_path_parent):
            #     os.makedirs(jpg_path_parent)
            frame_array = process_vidseq(movie_path)
            # success = frameArr_to_frameJpg(frame_array, jpg_path_parent)
            ucf101_frame_dict[instance_name] = frame_array[:cut_at_n_frame]
            print("Finish process {}".format(instance_name))
            instance_idx += 1
    return ucf101_frame_dict


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
