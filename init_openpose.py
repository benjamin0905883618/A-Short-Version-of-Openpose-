import sys
import cv2
import os
from sys import platform
import argparse
import time
import numpy as np

def initial_openpose():
    #import openpose
    dir_path = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(dir_path + '/python/openpose/Release');
    os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/x64/Release;' +  dir_path + '/bin;'
    import pyopenpose as op
    global op

    #parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_display",default = False,help = "Enable to disable the visual display")
    parser.add_argument("--num_gpu", default=op.get_gpu_number(), help="Number of GPUs.")
    args = parser.parse_known_args()

    #load model,if some flag needed,do it here
    params = dict()
    params["model_folder"] = "models/"
    params["num_gpu"] = int(vars(args[0])["num_gpu"])
    params["disable_blending"] = True
    numberGPUs = int(params["num_gpu"])

    #initial openpose
    global opWrapper
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    #global opWrapper
    opWrapper.start()

def transform_image(image):
    datums = []
    images = []
    datum = op.Datum()
    images.append(image)
    datum.cvInputData = images[-1]
    datums.append(datum)
    opWrapper.waitAndEmplace([datums[-1]])
    opWrapper.waitAndPop([datum])
    return datum.cvOutputData
