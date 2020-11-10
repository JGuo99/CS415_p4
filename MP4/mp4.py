import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import sys
import math 

def skinToneData(img):  
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  
    height, width, channel = hsv.shape
    freq = np.zeros((180, 256)) # Hue: 0 - 179 | Sat: 0 - 255 

    for i in range(height):
        for j in range(width):
            hue = hsv[i][j][0]
            sat = hsv[i][j][1]
            freq[hue][sat] += 1
    freq_norm = cv2.normalize(freq, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return freq_norm

def SkinDetect(freq, img):
    hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    height, width, channel = hsvImg.shape
    
    for i in range(height):
        for j in range(width):
            hue = hsvImg[i][j][0]
            sat = hsvImg[i][j][1]
            if freq[hue][sat] == 0:
                hsvImg[i][j][2] = 0
    hsvImg = cv2.cvtColor(hsvImg, cv2.COLOR_HSV2BGR)
    return hsvImg

def Histo2D(freq):
    fig, ax = plt.subplots(figsize = (10, 10))
    ax.imshow(
        freq, cmap = plt.cm.nipy_spectral,
        extent=[
            0, 180,
            0, 256
        ]
    )
    ax.set_title('2D Histogram of Trained Images')
    ax.set_xlabel('Hue')
    ax.set_ylabel('Saturation')

    plt.savefig('result.png')
    plt.show()

if __name__ == '__main__':
    # Training Images
    sampGun = cv2.imread('gun1_test.bmp')
    dataGun = skinToneData(sampGun)

    sampPoint = cv2.imread('pointer1_test.bmp')
    dataPoint = skinToneData(sampPoint)

    sampJoy = cv2.imread('joy1_test.bmp')
    dataJoy = skinToneData(sampJoy)

    sampOne = cv2.imread('skin1.bmp')
    dataOne = skinToneData(sampOne)

    sampTwo = cv2.imread('skin2.bmp')
    dataTwo = skinToneData(sampTwo)
    
    sampThree = cv2.imread('skin3.bmp')
    dataThree = skinToneData(sampThree)

    sampFour = cv2.imread('skin4.bmp')
    dataFour = skinToneData(sampFour)

    result = dataGun + dataPoint + dataJoy + dataOne + dataTwo + dataThree + dataFour
    Histo2D(result)

    # Testing Image     
    img = cv2.imread('gun1.bmp')
    final_gun = SkinDetect(result, img)

    img = cv2.imread('pointer1.bmp')
    final_pointer = SkinDetect(result, img)

    img = cv2.imread('joy1.bmp')
    final_joy = SkinDetect(result, img)

    # This does nothing to the logic or algorithm.
    # Just used to show images in one window.
    allImg = np.hstack((final_gun, final_pointer, final_joy))

    cv2.imwrite("allSkinImage.bmp", allImg)
    cv2.imwrite("gunSkinImage.bmp", final_gun)
    cv2.imwrite("pointerSkinImage.bmp", final_pointer)
    cv2.imwrite("joySkinImage.bmp", final_joy)
    cv2.imshow("Segmentation Image", allImg)
    cv2.waitKey(0)