import numpy as np
from PIL import Image

import caffe
import vis

# own libraries
import colorgram
import os
from os import listdir
from os.path import isfile, join
from pathlib import Path
import random
import shutil
import time
import datetime
import math
import sys

# own code - Jasper
PRINT_SECONDS = 0.2
REVIEW_SECONDS = 5
REST_SECONDS = 10
ERROR_ABOVE = "Image {height, width} has above 1000 pixels"
PASS_BELOW = "Image {height, width} has below 1000 pixels"
dm = "" # global dimensions string
# flow
"""
    1. Enter number of images to segment 
    2. check resume file
    3. read and get all images to list
    4. get index of file from resume and add 1 to index
    5. begin process of segment
    6. make csv in writing and listing the features
    7. move to next image from the list
    8. Loop back to 5

    THINGS NEED TO DO:
    1. write current painting file to resume - x
    2. reshape shape layer (dimensions of image) every segmentation - x
    3. Find algorithm to group the rgb colors to their (primary or secondary) family colors - x
        - Grouped colors according to the hue color wheel {Red, Yellow, Green, Cyan, Blue, Magenta}
    4. Export to csv
        - Features: {cname, cprop, }
        - Log everything first to a .txt file before exporting for debugging trace
"""


# functions
session_count = 0
def trackSession():
    global session_count
    if session_count == 0:
        session_count += 1
        return True
    else:
        return False
    return False

def setSession(file, flag=True):
    now = datetime.datetime.now()
    if flag:
        file.write("\n-------------------- SESSION - {} -------------------------\n".format(now.strftime("%Y-%m-%d %H:%M")))
    else:
        file.write("-------------------- SESSION - {} -------------------------\n".format(now.strftime("%Y-%m-%d %H:%M")))


def delayPrint(string, seconds): # n seconds delay printing
    time.sleep(seconds)
    exportLogs(string)
    print(string)

def getPaintings(path):
    paintings = [f.split(".")[0] for f in listdir(path) if isfile(join(path, f))]
    paintings.sort()
    return paintings

def exportLogs(logs, f="demo/logs.log"):
    logs += "\n"
    if(isfile(f)):
        file = open(f, "a")
        if trackSession():
            setSession(file)
        file.write(logs)
        file.close()
    else:
        print("Log file does not exist!")
        print("Creating {} file...".format(f))
        file = open(f, "a+")
        if trackSession():
            setSession(file)
        file.write(logs)
        file.close()

def createCurrentLog(fp):
    filepath = fp.split("/")
    f = "demo/output/{}.log".format(filepath[len(filepath) - 1].split(".")[0])
    if(isfile(f)):
        delayPrint("Resuming {} file".format(f), PRINT_SECONDS)
        file = open(f, "w")
        setSession(file, False)
        file.close()
    else:
        print("{} log file does not exist!".format(f))
        print("Creating {} file...".format(f))
        file = open(f, "w+")
        setSession(file, False)
        file.close()

def readResume(f="demo/resume.txt"):
    fp_resume = ""
    delayPrint("Checking {} file...".format(f), PRINT_SECONDS)
    if(isfile(f)):
        file = open(f, "r")
        delayPrint("Reading file...", PRINT_SECONDS)
        fp_resume = file.read()
        delayPrint("Last segmented image: {}".format(fp_resume.rstrip()), PRINT_SECONDS)
        delayPrint("Closing file...", PRINT_SECONDS)
        file.close()
    else:
        delayPrint("File does not exist!", PRINT_SECONDS)
    return fp_resume

def writeResume(current_painting_path, f="demo/resume.txt"):
    delayPrint("Checking {} file...".format(f), PRINT_SECONDS)
    if(isfile(f)):
        file = open(f, "w+")
        delayPrint("Saving last segmented image path...", PRINT_SECONDS)
        delayPrint("Writing file...", PRINT_SECONDS)
        file.write(current_painting_path)
        delayPrint("Closing file...", PRINT_SECONDS)
        file.close()
    else:
        delayPrint("File does not exist!", PRINT_SECONDS)

def writeErrorFile(current_painting_path, error="", f="demo/error.log"):
    delayPrint("Checking {} file...".format(f), PRINT_SECONDS)
    if(isfile(f)):
        file = open(f, "a")
        delayPrint("Writing error file...", PRINT_SECONDS)
        file.write("{} - {}\n".format(current_painting_path, error))
        delayPrint("Closing error file...", PRINT_SECONDS)
        file.close()
    else:
        delayPrint("{} log file does not exist!".format(f), PRINT_SECONDS)
        delayPrint("Creating {} file...".format(f), PRINT_SECONDS)
        delayPrint("Writing error file...", PRINT_SECONDS)
        file = open(f, "a+")
        file.write("{} - {}\n".format(current_painting_path, error))
        delayPrint("Closing error file...", PRINT_SECONDS)
        file.close()

def reshapeInputLayer(img, f="voc-fcn8s/test.prototxt"):
    delayPrint("Checking {} file...".format(f), PRINT_SECONDS)
    LINE_NUMBER = 7
    width, height = img.size
    if(isfile(f)):
        with open(f, "r") as file:
            delayPrint("Reading file...", PRINT_SECONDS)
            data = file.readlines()
            delayPrint("Reshaping input layer...", PRINT_SECONDS)
            data[LINE_NUMBER] = "    shape { dim: 1 dim: 3 dim: %s dim: %s }\n"%(height, width)
            delayPrint(data[LINE_NUMBER], PRINT_SECONDS)
        with open(f, "w+") as file:
            delayPrint("Writing file...", PRINT_SECONDS)
            file.writelines(data)
        delayPrint("Closing file...", PRINT_SECONDS)
        file.close()
    else:
        delayPrint("File does not exist!", PRINT_SECONDS)

def checkImageSize1000000(img):
    delayPrint("Checking if image {width, height} has 1,000,000 and above pixels...", PRINT_SECONDS)
    width, height = img.size
    dimensions = " - ({} x {})".format(height, width)
    global dm
    dm = dimensions
    ONE_MIL = 1000000
    delayPrint("Images (H x W) has dimensions {} x {}".format(height, width), PRINT_SECONDS)
    if((width * height) >= ONE_MIL):
        delayPrint(ERROR_ABOVE+dimensions, PRINT_SECONDS)
        return True
    else:
        delayPrint(PASS_BELOW+dimensions, PRINT_SECONDS)
        return False
    return False

def loop(paintings_path, paintings, current_painting):
    n = input("Enter number of images to segment: ")
    index = paintings.index(current_painting.split(".")[0])
    end = len(paintings) - 1
    last = index+int(n)
    if(index != 0):
        if(index == end):
            last = end
        else:
            index += 1
            last += 1
    for x in range(index, last):
        current_painting_path = paintings_path + "/" + paintings[x] + ".jpg"
        delayPrint(current_painting_path, PRINT_SECONDS)
        createCurrentLog(paintings[x])
        start_time = datetime.datetime.now()
        delayPrint("---------- Start Time - {:s} ----------".format(str(start_time)), PRINT_SECONDS)
        delayPrint("Index of painting: {}".format(x), PRINT_SECONDS)
        delayPrint("Last expected painting index: {}".format(last), PRINT_SECONDS)
        delayPrint("", REVIEW_SECONDS)
        try:
            segmentation(current_painting_path, paintings[x])
        except:
            error = sys.exc_info()[0]
            print("Error {} found! Writing error log.".format(error))
            writeErrorFile(current_painting_path, error)
        end_time = datetime.datetime.now()
        delayPrint("---------- End Time - {:s} ----------".format(str(end_time)), PRINT_SECONDS)
        elapsed_time = end_time - start_time
        delayPrint("Elapsed time is: {:s}".format(str(elapsed_time)), REVIEW_SECONDS)
        delayPrint("Index of painting: {}".format(x), PRINT_SECONDS)
        delayPrint("Last expected painting index: {}".format(last), PRINT_SECONDS)
        delayPrint("", REVIEW_SECONDS)
        if int(n) > 1 and x != last - 1: # rest (60 seconds) if images are greater to 1 and no rest if last image
            time.sleep(REST_SECONDS)
        # if x == last - 1:
        writeResume(current_painting_path)

def segmentation(path, current_painting):
    # the demo image is "2007_000129" from PASCAL VOC

    # load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
    # im = Image.open('demo/image.jpg')
    # path = "demo/Trials/twice.jpg"
    im = Image.open(path)
    if checkImageSize1000000(im):
        writeErrorFile(path, ERROR_ABOVE+dm)
    else:
        # reshape input layer from dimensions of image H x W
        reshapeInputLayer(im)
        delayPrint("Starting to segment the image -- {} -- in {} seconds".format(current_painting, REVIEW_SECONDS), PRINT_SECONDS)
        # delay for 5 seconds for reviewing of image name
        time.sleep(REVIEW_SECONDS)
        in_ = np.array(im, dtype=np.float32)
        # print(in_)
        in_ = in_[:,:,::-1]
        # time.sleep(120)
        in_ -= np.array((104.00698793,116.66876762,122.67891434))
        in_ = in_.transpose((2,0,1))

        # Own code:
        # Set mode to CPU since GPU can't handle much memory
        caffe.set_mode_cpu()
        # load net
        net = caffe.Net('voc-fcn8s/deploy.prototxt', 'voc-fcn8s/fcn8s-heavy-pascal.caffemodel', caffe.TEST)
        # shape for input (data blob is N x C x H x W), set data
        net.blobs['data'].reshape(1, *in_.shape)
        net.blobs['data'].data[...] = in_
        # run net and take argmax for prediction
        net.forward()
        out = net.blobs['score'].data[0].argmax(axis=0)

        # visualize segmentation in PASCAL VOC colors
        voc_palette = vis.make_palette(21)
        out_im = Image.fromarray(vis.color_seg(out, voc_palette))
        # out_im.save('demo/output.png')
        out_im.save('demo/output/output_%s.png'%(current_painting.split(".")[0]))
        logfile = "demo/output/"+current_painting+".log"
        masked_im = Image.fromarray(vis.vis_seg(im, out, voc_palette, 0.5, logfile))

        # print extracted colors of original image
        vis.extractColors(path, logfile)

        # masked_im.save('demo/visualization.jpg')
        masked_im.save('demo/output/output_%s.jpg'%(current_painting))
# end

# main process
delimiter = "/"
fp_resume = readResume()
current_painting = fp_resume.split("/")
current_painting = current_painting[len(current_painting) - 1].rstrip() # rstrip() for removing white spaces and new line instances
paintings_path = fp_resume.split("/")
paintings_path.remove(paintings_path[len(paintings_path) - 1])
paintings_path = delimiter.join(paintings_path)
paintings = getPaintings(paintings_path)
loop(paintings_path, paintings, current_painting)

