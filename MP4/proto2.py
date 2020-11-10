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
    freq_norm = cv2.normalize(freq, None, 0, 1000, cv2.NORM_MINMAX, cv2.CV_8U)
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
    return hsvImg

# She's not readyyyy!!
def Histo2D(freq, skinImg):
    fig, ax = plt.subplots(1, 2, figsize = (10, 10))
    ax[0].imshow(
        freq, cmap = plt.cm.nipy_spectral,
        extent=[
            0, 180,
            0, 256
        ]
    )
    ax[0].set_title('2D Histogram of HSV')
    ax[0].set_xlabel('Hue')
    ax[0].set_ylabel('Saturation')

    ax[1].imshow(skinImg)
    ax[1].set_title('Skin Detection')

    plt.show()

if __name__ == '__main__':
    # Training Images
    sampZero = cv2.imread('gun1_test.bmp')
    dataZero = skinToneData(sampZero)

    sampOne = cv2.imread('skin1.bmp')
    dataOne = skinToneData(sampOne)

    sampTwo = cv2.imread('skin2.bmp')
    dataTwo = skinToneData(sampTwo)
    
    sampThree = cv2.imread('skin3.bmp')
    dataThree = skinToneData(sampThree)

    sampFour = cv2.imread('skin4.bmp')
    dataFour = skinToneData(sampFour)

    sampFive = cv2.imread('skin5.bmp')
    dataFive = skinToneData(sampFive)

    result = dataZero + dataOne +  dataThree + dataFour + dataFive

    # Debug
    # np.savetxt("output.txt", result, fmt="%d")
    # print(result)

    # Testing Image     
    img = cv2.imread('custom.jpg')
    final = SkinDetect(result, img)
    final_plt = cv2.cvtColor(final, cv2.COLOR_HSV2RGB)  # HSV to RGB since we are using PLT to show
    final_cv = cv2.cvtColor(final, cv2.COLOR_HSV2BGR)  # HSV to RGB since we are using PLT to show

    cv2.imwrite("customTestImage.jpg", final_cv)
    cv2.imshow("Segmentation Image", final_cv)
    cv2.waitKey(0)
    Histo2D(result, final_plt)
    
    # Debugging
    # sampOne = cv2.imread('gun1_test.bmp', cv2.IMREAD_COLOR) # uint8
    # sampOne = cv2.imread('pointer1_test.bmp')
    # sampOne = cv2.imread('skin5.jpg') 
    # hsv1 = cv2.cvtColor(sampOne, cv2.COLOR_BGR2HSV)
    # dataOne = skinToneData(hsv1)    
    # cv2.imshow("Image", dataOne)
    # cv2.waitKey(0)
    
    # img = cv2.imread('gun1.bmp')
    # hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # combined = SkinDetect(dataOne, hsvImg)
    # combined = cv2.cvtColor(combined, cv2.COLOR_HSV2BGR)
    # cv2.imshow("HSV Image", combined)
    # cv2.waitKey(0)