import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os

from skimage import filters
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
from skimage import data, img_as_float
from skimage.util import random_noise
from skimage import feature
from skimage.filters import unsharp_mask
from skimage import measure

from scipy import misc #for preprocessing
import imageio
import os
import PIL  
from PIL import Image
from PIL import ImageOps

#disclaimer: Most code here is based of sklearn docs and was manipulated so that it could be used for the particular training task. https://scikit-image.org/docs/dev/auto_examples/
#Feature extracting and matching for EDA https://docs.opencv.org/3.3.1/d1/d89/tutorial_py_orb.html
# The preprocessing functions are entirely of my design and idea. such as the image removal function

def ImportantFeatures(img):
    # Initiate ORB detector
    orb = cv2.ORB_create()
    # find the keypoints with ORB
    kp = orb.detect(img,None)
    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)
    # draw only keypoints location,not size and orientation
    img2 = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
    plt.imshow(img2), plt.show()

"""
orb = cv2.ORB_create()

#Creates and identifies areas of interest for the particular cherry 
key_points, description = orb.detectAndCompute(img_cherry, None)
img_building_keypoints = cv2.drawKeypoints(img_cherry, 
                                           key_points, 
                                           img_cherry, 
                                           flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#Draw circles and display them.
plt.figure(figsize=(8, 8))
plt.title('ORB Interest Points')
plt.imshow(img_building_keypoints); plt.show()
"""

#as from seeing above the important features identified are not necessarily the ones we want it to be picking up so perhaps applying a filter
# will allow for better detection of certain features 

def filterImage(img1_name):
    
    fig, ax = plt.subplots(nrows=2, ncols=2)

    img1 = img1_name
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    edges = filters.sobel(img1)

    low = 0.1
    high = 0.35

    lowt = (edges > low).astype(int)
    hight = (edges > high).astype(int)
    hyst = filters.apply_hysteresis_threshold(edges, low, high)

    ax[0, 0].imshow(img1, cmap='gray')
    ax[0, 0].set_title('Original image')

    ax[0, 1].imshow(edges, cmap='magma')
    ax[0, 1].set_title('Sobel edges')

    ax[1, 0].imshow(lowt, cmap='magma')
    ax[1, 0].set_title('Low threshold')

    ax[1, 1].imshow(hight + hyst, cmap='magma')
    ax[1, 1].set_title('Hysteresis threshold')

    for a in ax.ravel():
        a.axis('off')

    plt.tight_layout()

    plt.show()

    return edges

def image_detect_and_compute(detector, img_name):
    """Detect and compute interest points and their descriptors."""
    img = cv2.imread(os.path.join(dataset_path_cherry, img_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    kp, des = detector.detectAndCompute(img, None)
    return img, kp, des
    

def draw_image_matches(detector, img1_name, img2_name, nmatches=10):
    """Draw ORB feature matches of the given two images."""
    img1, kp1, des1 = image_detect_and_compute(detector, img1_name)
    img2, kp2, des2 = image_detect_and_compute(detector, img2_name)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x: x.distance) # Sort matches by distance.  Best come first.
    
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:nmatches], img2, flags=2) # Show top 10 matches
    plt.figure(figsize=(8, 8))
    plt.title("Important feature matches")
    plt.imshow(img_matches); plt.show()

    
    
def HarrisCornerDetector(img):
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)
    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)
    # Threshold for an optimal value, it may vary depending on the image.
    img[dst>0.01*dst.max()]=[0,0,255]
    plt.title('Harris Corner Detector')
    plt.imshow(img)
    plt.show()


def Noise(img):
    img = img

    sigma = 0.155
    noisy = random_noise(img, var=sigma**2)

    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(8, 5),
                       sharex=True, sharey=True)

    plt.gray()

    # Estimate the average noise standard deviation across color channels.
    sigma_est = estimate_sigma(noisy, multichannel=True, average_sigmas=True)
    # Due to clipping in random_noise, the estimate will be a bit smaller than the
    # specified sigma.
    print(f"Estimated Gaussian noise standard deviation = {sigma_est}")

    ax[0, 0].imshow(noisy)
    ax[0, 0].axis('off')
    ax[0, 0].set_title('Noisy')
    ax[0, 1].imshow(denoise_tv_chambolle(noisy, weight=0.1, multichannel=True))
    ax[0, 1].axis('off')
    ax[0, 1].set_title('TV')
    ax[0, 2].imshow(denoise_bilateral(noisy, sigma_color=0.05, sigma_spatial=15,
                multichannel=True))
    ax[0, 2].axis('off')
    ax[0, 2].set_title('Bilateral')
    ax[0, 3].imshow(denoise_wavelet(noisy, multichannel=True))
    ax[0, 3].axis('off')
    ax[0, 3].set_title('Wavelet denoising')

    ax[1, 1].imshow(denoise_tv_chambolle(noisy, weight=0.2, multichannel=True))
    ax[1, 1].axis('off')
    ax[1, 1].set_title('(more) TV')
    ax[1, 2].imshow(denoise_bilateral(noisy, sigma_color=0.1, sigma_spatial=15,
                multichannel=True))
    ax[1, 2].axis('off')
    ax[1, 2].set_title('(more) Bilateral')
    ax[1, 3].imshow(denoise_wavelet(noisy, multichannel=True, convert2ycbcr=True
                                ))
    ax[1, 3].axis('off')
    ax[1, 3].set_title('Wavelet denoising\nin YCbCr colorspace')
    ax[1, 0].imshow(img)
    ax[1, 0].axis('off')
    ax[1, 0].set_title('Original')

    fig.tight_layout()

    plt.show()


def unsharp_masking(img):
    image = img
    result_1 = unsharp_mask(image, radius=1, amount=1)
    result_2 = unsharp_mask(image, radius=5, amount=2)
    result_3 = unsharp_mask(image, radius=20, amount=1)

    fig, axes = plt.subplots(nrows=2, ncols=2,
                         sharex=True, sharey=True, figsize=(7, 7))
    ax = axes.ravel()

    ax[0].imshow(image, cmap=plt.cm.gray)
    ax[0].set_title('Original image')
    ax[1].imshow(result_1, cmap=plt.cm.gray)
    ax[1].set_title('Enhanced image, radius=1, amount=1.0')
    ax[2].imshow(result_2, cmap=plt.cm.gray)
    ax[2].set_title('Enhanced image, radius=5, amount=2.0')
    ax[3].imshow(result_3, cmap=plt.cm.gray)
    ax[3].set_title('Enhanced image, radius=20, amount=1.0')

    for a in ax:
        a.axis('off')
    fig.tight_layout()
    plt.show()

def SIFT(img): #https://docs.opencv.org/3.3.1/da/df5/tutorial_py_sift_intro.html
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(gray,None)
    img=cv2.drawKeypoints(gray,kp,img)
    plt.title('SIFT')
    plt.imshow(img)
    plt.show()

def contour(img):

    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find contours at a constant value of 0.8
    contours = measure.find_contours(gray, 0.8)

    # Display the image and plot all contours found
    fig, ax = plt.subplots()
    ax.imshow(gray, cmap=plt.cm.gray)

    for n, contour in enumerate(contours):
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

def guassianFilter(img):
    new_img = cv2.GaussianBlur(img, (7, 7),0)
    plt.figure(figsize=(11,6))
    plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_HSV2RGB)),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(cv2.cvtColor(new_img, cv2.COLOR_HSV2RGB)),plt.title('Gaussian Filter')
    plt.xticks([]), plt.yticks([])
    plt.show()

def colour_histogram(img):
    chans = cv2.split(img)
    colors = ("b", "g", "r")
    plt.figure()
    plt.title("Color Histogram for a outlier image")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    features = []
 
    # loop over the image channels
    for (chan, color) in zip(chans, colors):
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        features.extend(hist)
        plt.plot(hist, color = color)
        plt.xlim([0, 256])

    plt.show()
 
def gray_histogram(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.plot(hist)
    plt.xlim([0, 256])
    plt.show()


def picture_to_arr(image):   
        r=g=b=0
        arr = imageio.imread(image)
        arr_list = arr.tolist()

        for row in arr_list:
            for col in row:
                p_b = col[0]
                p_g = col[1]
                p_r = col[2]
               
                if p_r > (p_g & p_b):
                    r+=1
                elif p_g > (p_b & p_r):
                    g+=1  
                elif p_b > (p_g & p_r):
                    b+=1
         
        minimum_req = 0.13 * (300 * 300) #pixels per image
        
        if r < minimum_req:
            print(image)
            print ("the of no* of red dominant pixels in this image=",r)
            print ("requirement is",minimum_req)
            os.remove(image)


def image_black(image):
    im = Image.open(image)
    pixels = im.getdata()
    black_thresh = 50
    nblack = 0
    for pixel in pixels:
        if pixel < black_thresh:
            nblack += 1
    n = len(pixels)

    if (nblack / float(n)) > 0.5:
        print("mostly black")

def resize_img():
    for filename in os.listdir('/Users/harryrodger/Desktop/data/strawberry/'):
        if filename.endswith('.jpg'):
            img = ('/Users/harryrodger/Desktop/data/strawberry/%s' % filename)
            OGimg = Image.open(img)
            size = (100,100)
            resized_image = ImageOps.fit(OGimg,size,Image.ANTIALIAS)
            resized_image.save(filename)

       




"""
resize_img()
print('Done')
"""
"""
#where the magic happens
dataset_path_cherry = '/Users/harryrodger/Desktop/data/cherry'
dataset_path_tomato = '/Users/harryrodger/Desktop/data/tomato'
dataset_path_strawberry = '/Users/harryrodger/Desktop/data/strawberry'

img_cherry = cv2.imread(os.path.join(dataset_path_cherry, 'cherry_0257.jpg'))
img_cherry = cv2.cvtColor(img_cherry, cv2.COLOR_BGR2RGB)

img_tomato = cv2.imread(os.path.join(dataset_path_tomato, 'tomato_0420.jpg'))
img_tomato = cv2.cvtColor(img_tomato, cv2.COLOR_BGR2RGB)

img_strawberry = cv2.imread(os.path.join(dataset_path_strawberry, 'strawberry_0434.jpg'))
img_strawberry = cv2.cvtColor(img_strawberry, cv2.COLOR_BGR2RGB)

#pick out the things which the detectors are finding the interesting things.
#EDA for cherry

#0257 cherry is a serious outlier



#finds the rgb values of an image and splits them
colour_histogram(img_cherry)
gray_histogram(img_cherry)


colour_histogram(img_tomato)
gray_histogram(img_tomato)


colour_histogram(img_strawberry)
gray_histogram(img_strawberry)


guassianFilter(img_cherry)

#preprocessing the images // removing ones which contain little red 

image_black('/Users/harryrodger/Desktop/data/tomato/tomato_0144.jpg')

picture_to_arr('/Users/harryrodger/Desktop/data/strawberry/strawberry_0227.jpg')
picture_to_arr('/Users/harryrodger/Desktop/data/strawberry/strawberry_1262.jpg')
picture_to_arr('/Users/harryrodger/Desktop/data/tomato/tomato_0144.jpg')
picture_to_arr('/Users/harryrodger/Desktop/data/tomato/tomato_0157.jpg')


"""

for filename in os.listdir('/Users/harryrodger/Desktop/data/tomato/'):
    picture_to_arr('/Users/harryrodger/Desktop/data/tomato/%s' % filename)


"""
for filename in os.listdir('/Users/harryrodger/Desktop/data/cherry/'):
    picture_to_arr('/Users/harryrodger/Desktop/data/cherry/%s' % filename)





for filename in os.listdir('/Users/harryrodger/Desktop/data/strawberry/'):
    if filename.endswith('.jpg'):
        picture_to_arr('/Users/harryrodger/Desktop/data/strawberry/%s' % filename)


Noise(img_cherry)

#ORB feature extraction

ImportantFeatures(img_cherry)


ImportantFeatures(img_tomato)


ImportantFeatures(img_strawberry)


orb = cv2.ORB_create()


draw_image_matches(orb, 'cherry_0020.jpg', 'cherry_0160.jpg')
sift = cv2.xfeatures2d.SIFT_create()
kp, des = sift.detectAndCompute(img_cherry, None)
img_kp = cv2.drawKeypoints(img_cherry, kp, img_cherry)

plt.figure(figsize=(7, 7))
plt.imshow(img_kp); plt.show()


#Filetering feature extraction

filterImage(img_strawberry)

filterImage(img_cherry)

filterImage(img_tomato)



HarrisCornerDetector(img_cherry)

contour(img_cherry)

unsharp_masking(img_cherry)

SIFT(img_cherry)
"""







