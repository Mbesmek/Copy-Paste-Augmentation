import cv2
import numpy as np
import xml.etree.ElementTree as ET
import argparse
import glob


def loadImage(path):
    img = cv2.imread(path)
    return img


def resizeImage(image, width, height):
    dim = (width, height)
    return cv2.resize(image, dim)


def rotateImage(image, angle):
    # image
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # rotate our image by 45 degrees around the center of the image
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


def pasteImage(l_img, s_img, x_offset, y_offset):
    y1, y2 = y_offset, y_offset + s_img.shape[0]  # s_img =uap.png
    x1, x2 = x_offset, x_offset + s_img.shape[1]

    alpha_s = s_img[:, :, 1] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        l_img[y1:y2, x1:x2, c] = (alpha_s * s_img[:, :, c] + alpha_l * l_img[y1:y2, x1:x2, c])
    return l_img


def readAnnotationsAndGetBbx(xmlPath):
    tree = ET.parse(xmlPath)
    root = tree.getroot()

    list_with_all_boxes = []

    for boxes in root.iter('object'):
        filename = root.find('filename').text

        ymin, xmin, ymax, xmax = None, None, None, None

        ymin = int(boxes.find("bndbox/ymin").text)
        xmin = int(boxes.find("bndbox/xmin").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        xmax = int(boxes.find("bndbox/xmax").text)

        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(list_with_single_boxes)
    return list_with_all_boxes


def controlOverLaping(bbxList, uXmin, uYmin, uXmax, uYmax):
    xState = False
    yState = False
    uWeight=uXmax-uXmin
    uHeigth=uYmax-uYmin
    for box in bbxList:
        xState = False
        yState = False
        bXmin = box[0]
        bYmin = box[1]
        bXmax = box[2]
        bYmax = box[3]

        bWeight = bXmax - bXmin
        bHeigth = bYmax - bYmin
        if uWeight < bWeight:
            if (bXmax >= uXmin >= bXmin) or (bXmax >= uXmax >= bXmin):
                xState = True
                break
        else:
            if (uXmin <= bXmax):
                xState = True
                break
        if uHeigth < bHeigth:
            if (bYmax >= uYmin >= bYmin) or (bYmax >= uYmax >= bYmin):
                yState = True
                break
        else:
            if (uYmin <= bYmax):
                yState = True
                break
    return xState, yState


def shiftingImage(uapStartX, uapStartY, uapEndX, uapEndY, stepX, stepY, img):
    imageHeight, imageWidth = img.shape[:2]
    uapWidth = uapEndX - uapStartX
    uapHeight = uapEndY - uapStartY

    if uapEndX + stepX > imageWidth:
        uapStartX = 1
        uapEndX = uapWidth+1
    else:
        uapStartX += stepX
        uapEndX += stepX
    if uapEndY + stepY > imageHeight:
        uapStartY = 1
        uapEndY = uapHeight +1
    else:
        uapEndY = stepY + uapEndY
        uapStartY = stepY + uapStartY
    return uapStartX, uapStartY, uapEndX, uapEndY


def main():

    #terminden path verilecek örnek : python main.py --image_path augmentationPath/ --dst_image_path resultAugmentation/ --image_pasted_path uap.png
    parser = argparse.ArgumentParser(description='In this code, .....')
    parser.add_argument('--image_path', '-i', type=str, required=True, help="Image and XML Path")
    parser.add_argument('--dst_image_path', '-d', type=str, required=True, help="Image and XML Destination Path")
    parser.add_argument('--image_pasted_path', '-img_c', type=str, required=True, help="Image for pasted")
    args = parser.parse_args()
    path = args.image_path
    dstimgpath = args.dst_image_path
    pastedImg = args.image_pasted_path

    imgs = glob.glob(path+"*.jpg")
    xmls = glob.glob(path+"*.xml")
    uap = loadImage(pastedImg)  # image for paste



    for i in range(len(imgs)):
        # xmlPath = (imagePath.strip(".jpg") + ".xml")
        img = loadImage(imgs[i])
        xml=xmls[i]


        # uap=rotateImage(uap,90)

        list_with_all_boxes = readAnnotationsAndGetBbx(xml)
        imageHeight, imageWidth = img.shape[:2]
        uapHeight, uapWidth = uap.shape[:2]

        uapStartX = (imageWidth // 2 - uapWidth // 2) - 1
        uapStartY = (imageHeight // 2 - uapHeight // 2) - 1

        uapEndX = uapStartX + uapWidth
        uapEndY = uapStartY + uapHeight

        xState = True
        yState = True


        while xState or yState:
            xState, yState = controlOverLaping(list_with_all_boxes, uapStartX, uapStartY, uapEndX, uapEndY)
            if xState == True:
                uapStartX, uapStartY, uapEndX, uapEndY = shiftingImage(uapStartX, uapStartY, uapEndX, uapEndY, 20, 0, img)
            if yState == True:
                uapStartX, uapStartY, uapEndX, uapEndY = shiftingImage(uapStartX, uapStartY, uapEndX, uapEndY, 0, 20, img)


        print(xState, yState)
        print(uapStartX, uapStartY, uapEndX, uapEndY)


        tree = ET.parse(xml)
        root = tree.getroot()
        xlobject = ET.Element("object")
        xlbname = ET.SubElement(xlobject, "name")
        xlbname.text = "UAP"
        pose = ET.SubElement(xlobject, "pose")
        pose.text = "Unspecified"
        truncate = ET.SubElement(xlobject, "truncated")
        truncate.text = "0"
        difficult = ET.SubElement(xlobject, "difficult")
        difficult.text = "0"
        bndbox = ET.SubElement(xlobject, "bndbox")
        tree = ET.ElementTree(root)
        xlxmin = ET.SubElement(bndbox, "xmin")
        xlxmin.text = str(int(uapStartX))
        xlymin = ET.SubElement(bndbox, "ymin")
        xlymin.text = str(int(uapStartY))
        xlxax = ET.SubElement(bndbox, "xmax")
        xlxax.text = str(int(uapEndX))
        xlymax = ET.SubElement(bndbox, "ymax")
        xlymax.text = str(int(uapEndY))
        root.append(xlobject)
        
        temp_data = dstimgpath + "img_Aug" + str(i) + ".xml"
        with open(temp_data, "w") as fh:
            tree = ET.ElementTree(root)
            print(ET.tostring(root))
            fh.write(ET.tostring(root).decode("utf-8"))

        pastedImage = pasteImage(img, uap, uapStartX, uapStartY)
        temp_data = dstimgpath + "img_Aug" + str(i) + ".jpg"
        cv2.imwrite(temp_data, pastedImage)

        # save a image using extension

main()
