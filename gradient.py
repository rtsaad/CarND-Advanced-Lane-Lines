import cv2
import numpy as np

## GRADIENT
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    o1, o2 = (1,0)
    if orient=='y':
        o1, o2 = (0,1)    
    sobel = cv2.Sobel(img, cv2.CV_64F, o1, o2, ksize=sobel_kernel)
    sobel = np.absolute(sobel)
    # Rescale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*sobel/np.max(sobel))
    # Create a mask of 1's where the scaled gradient magnitude 
    sbinary = np.zeros_like(scaled_sobel)
    # Apply threshold
    sbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1    
    return sbinary    

def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    # Calculate gradient magnitude
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the magnitude 
    maq = np.sqrt(np.power(sobelx,2) + np.power(sobely,2))
    # Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255*maq/np.max(maq))
    # Create a binary mask where mag thresholds are met
    sbinary = np.zeros_like(scaled_sobel)
    # Apply threshold
    sbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1 
    return sbinary

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    sobelx = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    sobely = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Compute the direction of the gradient 
    direction = np.arctan2(sobely, sobelx) 
    # Create a binary mask where direction thresholds are met
    sbinary = np.zeros_like(direction)
    # Apply threshold
    sbinary[(direction >= thresh[0]) & (direction <= thresh[1])] = 1 
    return sbinary

def hls_threshold(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    #threshold saturation
    s_channel = hls[:,:,2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1
    return s_binary

def hue_threshold(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    #get yellow and white color for hue
    h_channel = hsv[:,:,0]
    s_channel = hsv[:,:,1]
    v_channel = hsv[:,:,2]
    #get only yellow
    y_binary = np.zeros_like(h_channel)
    y_binary[((h_channel >= 0) & (h_channel <= 40))
             & ((s_channel >= 80) & (s_channel <= 255))
             & ((v_channel >= 200) & (v_channel <= 255))] = 1 #200 - 255
    #get only white
    w_binary = np.zeros_like(h_channel)
    w_binary[((h_channel >= 20) & (h_channel <= 255))
             & ((s_channel >= 0) & (s_channel <= 80))
             & ((v_channel >= 200) & (v_channel <= 255))] = 1
    combined = np.zeros_like(h_channel)
    combined[((y_binary == 1) | (w_binary == 1))] = 1
    return combined

def test_gradient(img, ksize=3, thresh=(70, 255), direction_thresh=(0.7, 1.3)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(gray, orient='x', sobel_kernel=3, thresh=thresh)
    plt.imshow(gradx, cmap='gray')
    plt.show()
    grady = abs_sobel_thresh(gray, orient='y', sobel_kernel=3, thresh=thresh)
    plt.imshow(grady, cmap='gray')
    plt.show()
    mag_binary = mag_thresh(gray, sobel_kernel=3, thresh=thresh)
    plt.imshow(mag_binary, cmap='gray')
    plt.show()
    dir_binary = dir_threshold(gray, sobel_kernel=15, thresh=direction_thresh)
    plt.imshow(dir_binary, cmap='gray')
    plt.show()
    hls = hls_threshold(img, thresh=(170, 255))
    plt.imshow(hls, cmap='gray')
    plt.show()
    hue = hue_threshold(img)
    plt.imshow(hue, cmap='gray')
    plt.show()
    combined = np.zeros_like(dir_binary)    
    combined[(gradx == 1) | (hls == 1) | (hue == 1)] = 1
    #combined[((hls == 1) | (hue == 1))] = 1
    #combined[(gradx == 1) | (hls == 1)] = 1
    return combined

def combined_gradient(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gradx = abs_sobel_thresh(gray, orient='x', sobel_kernel=3, thresh=(20,100))
    hls = hls_threshold(img, thresh=(170, 255))
    hue = hue_threshold(img)
    combined = np.zeros_like(gradx)  
    combined[((hue == 1) & (hls == 1)) | (gradx == 1) ] = 1
    return combined

if __name__ == "__main__":
    import matplotlib.image as mpimg
    #Project imports
    import camera
    
    print("Test Gradient")
    img = mpimg.imread('test_images/straight_lines2.jpg')
    #img = mpimg.imread('test_images/test5.jpg')
    undistorted = camera.undistort_image(img)      
    warp = camera.warp(undistorted)
    gradient = combined_gradient(warp)
    camera.show_images(warp, gradient, title1='Original', title2='Gradient', filename='gradient_warp')
