import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import sys
import math 

    # Training Image
def skinToneData(img):    
    hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Shape of the HSV Image
    height = hsvImg.shape[0]
    width = hsvImg.shape[1]

    freq = np.zeros((180, 256))   # HSV Range: Hue[0, 179], Sat[0, 255]

    for i in range(width):
        for j in range(height):
            H = hsvImg[j][i][0]
            S = hsvImg[j][i][1]
            freq[H, S] += 1 
    freq = cv2.normalize(freq, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return freq

def SkinDetect(freq, img):
    height = img.shape[0]
    width = img.shape[1]

    for i in range(width):
        for j in range(height):
            hue = img[j][i][0]
            sat = img[j][i][1]
            # if freq[sat][hue] == 0:
            #     img[j][i][2] = 0
    return img

def Histo(h, s, freq):
    # Hthres_Min = np.min(Hthres)
    # Hthres_Max = np.max(Hthres)

    # Sthres_Min = np.min(Sthres)
    # Sthres_Max = np.max(Sthres)

    # hBin = np.linspace(Hthres_Min, Hthres_Max, 10)
    # sBin = np.linspace(Sthres_Min, Sthres_Max, 10)

    fig, ax = plt.subplots(figsize=(15, 10))

    # Histogram
    plt.hist2d(h, s, bins=freq, cmap = plt.cm.nipy_spectral)
    # ax.imshow(
    #     freq, cmap=plt.cm.nipy_spectral,
    #     extent=[
    #         h, s
    #     ]
    # )

    ax.set_title('2D Histogram')
    ax.set_xlabel('Hue')
    ax.set_ylabel('Saturation')

    fig.tight_layout(pad=3.0)
    # plt.savefig('result.png')
    plt.show()

if __name__ == '__main__':
    # Training Images
    light = cv2.imread('gun1_test.bmp')
    pink = cv2.imread('joy1_test.bmp')
    medium = cv2.imread('pointer1_test.bmp')
    brown = cv2.imread('skin4.jpg')
    dark = cv2.imread('skin5.jpg')
    
    dataOne = skinToneData(light)
    dataTwo = skinToneData(pink)
    dataThree = skinToneData(medium)
    dataFour = skinToneData(brown)
    dataFive = skinToneData(dark)

    result = np.array([dataOne, dataTwo, dataThree])

    print(result)

    # print("Data One: ", dataOne, "\nData Two: ", dataTwo)

    # Testing Image     
    img = cv2.imread('pointer1.bmp')
    hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    final = SkinDetect(result, hsvImg)
    h, s, v = cv2.split(hsvImg)
    Histo(h, s, result)

   
    # cv2.imshow("HSV Image", final)
    # cv2.waitKey(0)