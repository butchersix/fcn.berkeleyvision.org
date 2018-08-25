import os
from os import listdir
from os.path import isfile, join
from pathlib import Path
from PIL import Image
import time
import datetime

session_count = 0
PRINT_SECONDS = 0

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

def delayPrint(string, seconds, f=""): # n seconds delay printing
    time.sleep(seconds)
    exportLogs(string, "demo/resized_logs_random_samples.log")
    print(string)

def exportLogs(logs, f="demo/resized_logs_test.log"):
    logs += "\n"
    if(isfile(f)):
        file = open(f, "a")
        if trackSession():
            setSession(file, False)
        file.writelines(logs)
        file.close()
    else:
        print("Log file does not exist!")
        print("Creating {} file...".format(f))
        file = open(f, "a+")
        if trackSession():
            setSession(file, False)
        file.writelines(logs)
        file.close()

def getPaintings(path):
    paintings = [f for f in listdir(path) if isfile(join(path, f))]
    paintings.sort()
    return paintings

# resize image to 50 % scale
def resize(path, paintings):
    i = 1 # counter for the index of image
    for x in paintings:
        j = 0 # counter for how many scalings it took
        image_path = path+x
        delayPrint("{}. Resizing image {}".format(i, x), PRINT_SECONDS, getUpperPath(path))
        im = Image.open(image_path)
        height, width = im.size
        print("{} x {}".format(round(height/2), round(width/2)))
        resized_scale = (int(round(height/2)), int(round(width/2)))
        rh, rw = resized_scale
        j += 1
        while((rh*rw) >= 1000000): # this checks if total number of pixels is greater than or equal to 1 million
            rh, rw = (int(round(rh/2)), int(round(rw/2)))
            resized_scale = (rh, rw)
            j += 1
        im = im.resize((resized_scale), Image.ANTIALIAS)
        delayPrint("Final dimension size: {}".format(resized_scale), PRINT_SECONDS, getUpperPath(path))
        delayPrint("Number of scales: {}\n".format(j), PRINT_SECONDS, getUpperPath(path))
        output_dir = makeOutputDir(path)
        im.save(output_dir+x)
        i += 1

def makeOutputDir(path):
    # Make a new directory for the new dataset
    path = getUpperPath(path)
    output_dir = path+"_resized/"
    if not os.path.exists(output_dir):
        delayPrint("Creating directory", PRINT_SECONDS, getUpperPath(path))
        os.makedirs(output_dir)
        delayPrint("Directory created\n", PRINT_SECONDS, getUpperPath(path))
    return output_dir

def getUpperPath(path):
    delimiter = "/"
    path_splits = path.split("/")
    path_splits.remove(path_splits[len(path_splits) - 1])
    path = delimiter.join(path_splits)
    return path


# main process
paintings_path = input("Enter painting path dataset: ")
paintings = getPaintings(paintings_path)
resize(paintings_path, paintings)
