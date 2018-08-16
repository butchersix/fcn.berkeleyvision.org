import numpy as np

# own libraries imported
import colorgram
import math
import colorsys
import os
from os import listdir
from os.path import isfile, join
from pathlib import Path
import time
# end

# own code - Jasper
class_names = ["background", "aeroplane", "bicycle", "bird",
               "boat", "bottle", "bus", "car",
               "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person",
               "pottedplant", "sheep", "sofa", "train",
               "tvmonitor"]
image_path = ""
PRINT_SECONDS = 0.2
# end

def make_palette(num_classes):
    """
    Maps classes to colors in the style of PASCAL VOC.
    Close values are mapped to far colors for segmentation visualization.
    See http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit

    Takes:
        num_classes: the number of classes
    Gives:
        palette: the colormap as a k x 3 array of RGB colors
    """
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    for k in range(0, num_classes):
        label = k
        i = 0
        while label:
            palette[k, 0] |= (((label >> 0) & 1) << (7 - i))
            palette[k, 1] |= (((label >> 1) & 1) << (7 - i))
            palette[k, 2] |= (((label >> 2) & 1) << (7 - i))
            label >>= 3
            i += 1
    # own code - Jasper
    # print("make_palette():\n")
    # print(palette)
    # end
    return palette

def color_seg(seg, palette):
    """
    Replace classes with their colors.

    Takes:
        seg: H x W segmentation image of class IDs
    Gives:
        H x W x 3 image of class colors
    """
    return palette[seg.flat].reshape(seg.shape + (3,))


def vis_seg(img, seg, palette, alpha=0.5, logfile="unknown.log"):
    """
    Visualize segmentation as an overlay on the image.

    Takes:
        img: H x W x 3 image in [0, 255]
        seg: H x W segmentation image of class IDs
        palette: K x 3 colormap for all classes
        alpha: opacity of the segmentation in [0, 1]
    Gives:
        H x W x 3 image with overlaid segmentation
    """
    vis = np.array(img, dtype=np.float32)
    mask = seg > 0
    vis[mask] *= 1. - alpha
    vis[mask] += alpha * palette[seg[mask].flat]
    vis = vis.astype(np.uint8)

    # own code - Jasper
    total_pixels = totalNumPixels(seg, palette)
    # print("color_seg():\n")
    # print(palette)
    # print("Seg: \n")
    # print(seg)
    # print("Seg.flat:\n")
    # print(seg.flat)
    # print("Palette seg.flat:\n")
    # print(palette[seg.flat])
    exportLogs("Number of pixels: {:d}".format(total_pixels))

    # classes of pixels in tuple form
    exportLogs("Extract Unique Pixel Classes:\n")
    pixel_classes = [tuple(row) for row in palette[seg.flat]]

    # remove duplicate classes of pixels
    unique_classes = np.unique(pixel_classes, axis=0)

    # print result of pixel classes present in image
    for x in range(0, len(unique_classes.tolist())):
        exportLogs("{}. {}".format(x + 1, unique_classes.tolist()[x]))

    # determine index of class and relate it with class_names
    # numpy array must be converted to list for list manipulation
    exportLogs("---------- Class Names - RGB Value ----------")
    for i in unique_classes.tolist():
        # get index of pixel class from palette
        class_index = palette.tolist().index(i)

        # RGB value of pixel class
        class_color = tuple(i)

        # get the class name from class_names list based on PASCAL VOC list of classes         
        class_name = class_names[class_index]

        # compute region percentage of class from the image
        value = (pixel_classes.count(tuple(i))/total_pixels)
        percentage = value * 100
        
        # print results
        exportLogs("Class ID: {:d}".format(class_index))
        exportLogs("Class Color: {}".format(class_color))
        exportLogs("Class Name: {:s}".format(class_name))
        exportLogs("Percentage of region: {:.3f}%".format(percentage))
        exportLogs("C{:d} - {:s}: {}".format(class_index, class_name, value), logfile, False)
        exportLogs("\n")
        # exportLogs("\n", logfile, False)
    # end
    return vis

# own code - Jasper

# functions
def totalNumPixels(seg, palette):
    return len(palette[seg.flat])

def extractColors(image, logfile="unknown.log"):
    # extract 255 ^ 3 colors from image
    colors = colorgram.extract(image, 256**3)

    # list of rgb and hsv colors present in image and its proportion in the image
    color_classes = []

    # hue color counts and proportions
    red_c = yellow_c = green_c = cyan_c = blue_c = magenta_c = 0
    red_p = yellow_p = green_p = cyan_p = blue_p = magenta_p = 0
    for color in colors:
        red = color.rgb.r
        green = color.rgb.g
        blue = color.rgb.b
        h, s, v = colorsys.rgb_to_hsv(red, green, blue)
        h *= 360
        color_name = ""

        # equivalent values of hsv in degrees
        if (h >= 0 and h < 60) or h == 360:
            red_c += 1
            red_p += color.proportion
            color_name = "red"
        elif h >= 60 and h < 120:
            yellow_c += 1
            yellow_p += color.proportion
            color_name = "yellow"
        elif h >= 120 and h < 180:
            green_c += 1
            green_p += color.proportion
            color_name = "green"
        elif h >= 180 and h < 240:
            cyan_c += 1
            cyan_p += color.proportion
            color_name = "cyan"
        elif h >= 240 and h < 300:
            blue_c += 1
            blue_p += color.proportion
            color_name = "blue"
        elif h >= 300 and h < 360:
            magenta_c += 1
            magenta_p += color.proportion
            color_name = "magenta"
        
        color_classes.append("RGB Color - HSV - Proportion - Hue Color: {} - {} - {}% - {}".format(color.rgb, (h, s, v), (color.proportion * 100), color_name))
    
    # remove duplicates
    color_classes = set(color_classes)
    
    # show RGB colors and Proportions
    exportLogs("---------- Colors Present In Image ----------\n")
    for color in color_classes:
        exportLogs(color)
    
    # show list of hue colors
    exportLogs("\n\n---------- Hue Colors - Hue Proportions ----------\n")
    hue_colors = ["Red    ", "Yellow ", "Green ", "Cyan   ", "Blue   ", "Magenta"] # 7 spaces
    hue_colors1 = ["Red", "Yellow", "Green", "Cyan", "Blue", "Magenta"] # 7 spaces
    hue_color_count = [red_c, yellow_c, green_c, cyan_c, blue_c, magenta_c]
    hue_color_proportion = [red_p, yellow_p, green_p, cyan_p, blue_p, magenta_p]
    exportLogs("Color\t -\t Color Proportion\t -\t No. of instances")
    for x in range(0, len(hue_colors)):
        exportLogs("{}\t -\t {:.3f}%\t\t\t -\t {}".format(hue_colors[x], hue_color_proportion[x] * 100, hue_color_count[x]))
        exportLogs("{}: {} - {}".format(hue_colors1[x], hue_color_proportion[x], hue_color_count[x]), logfile, False)

    # show total number of colors present in the image
    exportLogs("--------------------------------------------------")
    exportLogs("Total number of colors: {}\n".format(len(color_classes)))

def delayPrint(string, seconds): # n seconds delay printing
    time.sleep(seconds)
    print(string)
    exportLogs(string)

def exportLogs(logs, f="demo/logs.log", flag=True):
    if flag:
        print(logs)
    logs += "\n"
    if(isfile(f)):
        file = open(f, "a")
        file.write(logs)
        file.close()
    else:
        print("Log file does not exist!")
        print("Creating {} file...".format(f))
        file = open(f, "a+")
        file.write(logs)
        file.close()

# function not used
def rgb_to_hsv(r, g, b):
    r = float(r)
    g = float(g)
    b = float(b)
    high = max(r, g, b)
    low = min(r, g, b)
    h, s, v = high, high, high

    d = high - low
    s = 0 if high == 0 else d/high

    if high == low:
        h = 0.0
    else:
        h = {
            r: (g - b) / d + (6 if g < b else 0),
            g: (b - r) / d + 2,
            b: (r - g) / d + 4,
        }[high]
        h /= 6
    return h, s, v
# end