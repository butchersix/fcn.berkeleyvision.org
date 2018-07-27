import numpy as np

# own libraries imported
import colorgram
# end

# own code - Jasper
class_names = ["background", "aeroplane", "bicycle", "bird",
               "boat", "bottle", "bus", "car",
               "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person",
               "pottedplant", "sheep", "sofa", "train",
               "tvmonitor"]
image_path = ""
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
    print("make_palette():\n")
    print(palette)
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


def vis_seg(img, seg, palette, alpha=0.5):
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
    print("color_seg():\n")
    print(palette)
    print("Seg: \n")
    print(seg)
    print("Seg.flat:\n")
    print(seg.flat)
    print("Palette seg.flat:\n")
    print(palette[seg.flat])
    print("Number of pixels: {:d}".format(total_pixels))

    # classes of pixels in tuple form
    print("Extract Unique Pixel Classes:\n")
    pixel_classes = [tuple(row) for row in palette[seg.flat]]

    # remove duplicate classes of pixels
    unique_classes = np.unique(pixel_classes, axis=0)

    # print result of pixel classes present in image
    print(unique_classes)

    # determine index of class and relate it with class_names
    # numpy array must be converted to list for list manipulation
    print("---------- Class Names - RGB Value ----------")
    for i in unique_classes.tolist():
        # get index of pixel class from palette
        class_index = palette.tolist().index(i)

        # RGB value of pixel class
        class_color = tuple(i)

        # get the class name from class_names list based on PASCAL VOC list of classes         
        class_name = class_names[class_index]

        # compute region percentage of class from the image
        percentage = (pixel_classes.count(tuple(i))/total_pixels) * 100
        
        # print results
        print("Class ID: {:d}".format(class_index))
        print("Class Color: {}".format(class_color))
        print("Class Name: {:s}".format(class_name))
        print("Percentage of region: {:.3f}%".format(percentage))
        print("\n")
    
    # show RGB colors and Proportions
    print("RGB Colors and Proportions: {}".format(extractColors(img)))
    # end
    return vis

# own code - Jasper

# functions
def totalNumPixels(seg, palette):
    return len(palette[seg.flat])

def setImagePath(path):
    image_path = path

def extractColors(image):
    # extract 255 colors from image
    colors = colorgram.extract(image, 10)

    # list of rgb colors present in image and its proportion in the image
    color_classes = []
    for color in colors:
        color_classes.append("RGB Color - Proportion: {} - {}".format(color.rgb, color.proportion))
    
    # remove duplicates
    color_classes = set(color_classes)
    return color_classes
# end