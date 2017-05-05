#Pipeline
# 1. Camera calibration
# 2. Distortion
# 3. Color Gradient
# 4. Perspective Transform

###
# 5. Detect lane lines
# 2. Lane Curve and offset from center

import os
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


#CONSTANTS
CAMERA_PARAMETERS_FILE = "parameter_camera.pkl"
WARP_PARAMETERS_FILE = "parameter_warp.pkl"

#COMMON FUNCTIONS
def show_image(img):
    if len(img.shape) > 2:
        plt.imshow(img)
    else:
        #Gray image
        plt.imshow(img, cmap='gray')
    plt.show()

def show_images(origin, undistorted, title1="", title2="", filename="image.png"):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    if len(origin.shape) > 2:
        ax1.imshow(origin)
    else:
        #Gray image
        ax1.imshow(origin, cmap='gray')        
    ax1.set_title(title1, fontsize=50)
    if len(undistorted.shape) > 2:
        ax2.imshow(undistorted)
    else:
        #Gray image
        ax2.imshow(undistorted, cmap='gray')
    ax2.set_title(title2, fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.savefig(filename, dpi=100)
    plt.show()

#Classes
# Define a class to receive the characteristics of each line detection
class Line():

    n = 5 # 1 second
    def __init__(self, width, height):
        self.reset(width, height)

    def reset(self, width, height):
        #numer of miss
        self.miss = 0
        #const ploty        
        self.ploty = np.linspace(0, height-1, height)
        # was the line detected in the last iteration?
        self.detected = False  #
        # x values of the last n fits of the line
        self.recent_xfitted = []# 
        #average x values of the fitted line over the last n iterations
        self.bestx = None#     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None#
        self.radius_of_curvature_n = []#
        #distance in meters of vehicle center from the line
        self.line_base_pos = None#
        self.line_base_pos_n = []#
        self.line_base_pos_center = None#
        self.line_base_pos_center_n = []# 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
        #confidence
        self.condidence = None


    def update(self, fit, fitx, values_x, values_y, center, radius, distance, confidence, new=False):
        
        self.detected = True

        if len(self.recent_xfitted) > self.n:
            self.recent_xfitted.pop()
        self.recent_xfitted.append(fitx)
        
        if len(self.radius_of_curvature_n) > self.n:
            self.radius_of_curvature_n.pop()
        self.radius_of_curvature_n.append(radius)

        if len(self.line_base_pos_n) > self.n:
            self.line_base_pos_n.pop()
        self.line_base_pos_n.append(distance)

        if len(self.line_base_pos_center_n) > self.n:
            self.line_base_pos_center_n.pop()
        self.line_base_pos_center_n.append(center)        

        self.allx = values_x
        self.ally = values_y
        
        self.current_fit = fit        
        self.confidence = confidence        
        
        #update moving average
        self.line_base_pos = np.average(self.line_base_pos_n)
        self.line_base_center_pos = np.average(self.line_base_pos_center_n)
        self.radius_of_curvature = np.average(self.radius_of_curvature_n)
        #self.bestx = np.average(self.recent_xfitted, axis=0)                      
        self.best_fit = np.polyfit(self.ally, self.allx, 2)
        self.bestx = self.best_fit[0]*self.ploty**2 + self.best_fit[1]*self.ploty + self.best_fit[2]


LEFT_LINE  = Line(1280, 720)
RIGHT_LINE = Line(1280, 720)
     
## CALIBRATE CAMERA
ret, mtx, dist, rvecs, tvecs = (None, None, None, None, None)

# Calibrate camera using the OpenCv chessboad method
def calibrate_camera(folder, nx=9, ny=6, show_corners=False):
    global ret, mtx, dist, rvecs, tvecs
    # prepare object points
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)
    objpoints = []
    imgpoints = []    

    print("Calibrate Camera")
    #iterate calibration images
    for fname in os.listdir(folder):
        print(fname)
        img = cv2.imread(folder + '/' + fname)        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)    
        # If found, draw corners
        if ret == True:            
            #Append corners and object
            objpoints.append(objp)
            imgpoints.append(corners)
            if show_corners:
                # Draw and display the corners
                cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
                plt.imshow(img)
                plt.show()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print("Calibration Finished")
    print("Save parameters")
    output = open(CAMERA_PARAMETERS_FILE, 'wb')
    #pickle.dump({ret: ret, mtx: mtx, dist: dist, rvecs: rvecs, tvecs: tvecs}, output)
    pickle.dump((ret, mtx, dist, rvecs, tvecs), output)
    output.close()    

## UNDISTORT IMAGE
# Undistort images with paramters found with the chessboard method
def undistort_image(img):
    global ret, mtx, dist, rvecs, tvecs
    if mtx is None or dist is None:
        #try to load from file
        try:
            camera_pickle = pickle.load( open( CAMERA_PARAMETERS_FILE, "rb" ) )
            (ret, mtx, dist, rvecs, tvecs) = camera_pickle
        except:
            calibrate_camera('camera_cal')
    return cv2.undistort(img, mtx, dist, None, mtx)
    

##Transform Bird View Perspective
M = None
Minv = None
def calibrate_warp_dashboard(img, nx=9, ny=6, show_corners=False):
    print("Start warp calibration")
    global M, Minv
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_size = (gray.shape[1], gray.shape[0])    
    #Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)
    if ret == True:
        # Source points
        src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
        # Destination points 
        offset = 100
        dst = np.float32([[offset, offset], [img_size[0]-offset, offset], 
                          [img_size[0]-offset, img_size[1]-offset], 
                          [offset, img_size[1]-offset]])
        # getPerspectiveTransform() to get M, the transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        print("Save parameters")
        output = open(WARP_PARAMETERS_FILE, 'wb')
        pickle.dump((M, Minv), output)
        output.close()
        print("Warp calibration Finished")
        if show_corners:
                # Draw and display the corners
                cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
                warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
                plt.imshow(warped)
                plt.show()

def calibrate_warp_lane(img, show_points=False):
    print("Start warp calibration")
    global M, Minv
    print(img.shape)
    img_size = (img.shape[1], img.shape[0])
    # Source points   
    src = np.float32([
        [556, 476], #[543, 486],
        [730, 476], #[738, 481],
        [1060, 690],#[1042, 676],
        [259, 690]  #[256, 690],
    ])
    # Destination points 
    offset_X = 1058
    offset_Y = 704
    dst = np.float32([
        [259, 300],#[247, 300], #280,  210
        [1059, 300],#[1059, 300],  #1000, 210
        [1059, 720],#[1059, 720],#1000, 676 
        [259, 720]#[247,  720] #280,  690
    ])
    # getPerspectiveTransform() to get M, the transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    print("Save parameters")
    output = open(WARP_PARAMETERS_FILE, 'wb')
    pickle.dump((M, Minv), output)
    output.close()
    print("Warp calibration Finished")
    if show_points:
        plt.imshow(img)
        plt.plot(src[0][0], src[0][1], '.')
        plt.plot(src[1][0], src[1][1], '.')
        plt.plot(src[2][0], src[2][1], '.')
        plt.plot(src[3][0], src[3][1], '.')
        plt.show()
        warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
        plt.imshow(warped)
        plt.plot(dst[0][0], dst[0][1], '.')
        plt.plot(dst[1][0], dst[1][1], '.')
        plt.plot(dst[2][0], dst[2][1], '.')
        plt.plot(dst[3][0], dst[3][1], '.')
        plt.show()        

def warp(img, nx=9, ny=6):
    global M, Minv
    if M is None:
        try:
            warp_pickle = pickle.load( open(WARP_PARAMETERS_FILE, "rb" ) )
            (M, Minv) = warp_pickle
        except:
            calibrate_warp(img, nx, ny)
    img_size = (img.shape[1], img.shape[0])    
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped

## Separate Region
def region_of_interest(img):
    vertices = np.array([[(0,img.shape[0]),(580,420),(690, 420),(img.shape[1], img.shape[0])]], dtype=np.int32)
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
    

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
             & ((v_channel >= 200) & (v_channel <= 255))] = 1
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
    #combined[(gradx == 1) | (hls == 1) | (hue == 1)] = 1
    #combined[((hls == 1) | (hue == 1))] = 1
    combined[(gradx == 1) | (hls == 1)] = 1
    return combined

def combined_gradient(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gradx = abs_sobel_thresh(gray, orient='x', sobel_kernel=3, thresh=(20,100))
    mag_binary = mag_thresh(gray, sobel_kernel=3, thresh=(50,100))
    dir_binary = dir_threshold(gray, sobel_kernel=15, thresh=(0.7, 1.3))
    hls = hls_threshold(img, thresh=(170, 255))
    hue = hue_threshold(img)
    combined = np.zeros_like(gradx)    
    combined[(gradx == 1) | ((mag_binary==1) & (dir_binary==1)) | (hls == 1) | (hue == 1)] = 1
    #combined[(hls == 1) | (hue == 1)] = 1
    return combined

#Fill Corner (polynomial)
def histogram_max(img_binary):
    histogram = np.sum(img_binary[img_binary.shape[0]//2:,:], axis=0)   
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.max(histogram[:midpoint])
    rightx_base = np.max(histogram[midpoint:])
    leftx_pos = np.argmax(histogram[:midpoint])
    rightx_pos = np.argmax(histogram[midpoint:])
    return (leftx_base,rightx_base, leftx_pos, rightx_pos)
    

def histogram_gradient(img_binary):
    histogram = np.sum(img_binary[img_binary.shape[0]//2:,:], axis=0)
    plt.plot(histogram)
    plt.show()

def plot_curve(img_binary, left_fitx, right_fitx, left_lane_inds=[], right_lane_inds=[]):
    
    out_img = np.dstack((img_binary, img_binary, img_binary))*255
    nonzero = img_binary.nonzero()    
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    ploty = np.linspace(0, img_binary.shape[0]-1, img_binary.shape[0] )
    
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.show()

def compute_metrics(p, lx, rx, width):    
    
    #ploty = np.linspace(0, 719, num=720)
    
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # 30/720 meters per pixel in y dimension
    xm_per_pix = 3.7/800 # 3.7/700 meters per pixel in x dimension
    lx = lx[::-1]  # Reverse to match top-to-bottom in y
    rx = rx[::-1]  # Reverse to match top-to-bottom in y
        
    y_eval = np.max(p)#np.concatenate((lefty,righty), axis=0))    

    # Fit new polynomials to x,y in world space    
    left_fit_cr = np.polyfit(p*ym_per_pix, lx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(p*ym_per_pix, rx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters

     #Compute center
    center_lane = (lx[0] + (rx[0] - lx[0])/2)
    center_img  = width/2
    center = abs((center_lane - center_img)*(3.7/800))    

    #Compute left
    left = abs((center_img - lx[0])*(3.7/800))

    #Compute right
    right = abs((rx[0] - center_img)*(3.7/800))
    
    return (left_curverad, right_curverad, center, left, right)

def sliding_window(img_binary):

    left_fit  = None
    right_fit = None
    left_fitx = None
    right_fitx= None
    left_lane_inds  = None
    right_lane_inds = None
    leftx = None
    rightx = None

    # Take a histogram of the bottom half of the image
    histogram = np.sum(img_binary[img_binary.shape[0]//2:,:], axis=0)    
    # Create an output image to draw on and  visualize the result
    #out_img = np.dstack((img_binary, img_binary, img_binary))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(img_binary.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img_binary.nonzero()    
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img_binary.shape[0] - (window+1)*window_height
        win_y_high = img_binary.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        #cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        #cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    try:
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
    except:
        return (left_fit, right_fit, left_fitx, right_fitx, left_lane_inds, right_lane_inds, leftx, rightx)

    # Generate x and y values for plotting
    ploty = np.linspace(0, img_binary.shape[0]-1, img_binary.shape[0] )    
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    #get radius
    (left_curverad, right_curverad) = (0,0)#compute_radius(ploty, left_fitx, lefty, right_fitx, righty)

    return (left_fit, right_fit, left_fitx, right_fitx, left_lane_inds, right_lane_inds, leftx, lefty, rightx, righty)

def fit_again(img_binary, left_fit, right_fit):

    #(a,b) = histogram_max(img_binary)
    #if a < 30 or b < 30:
    #    return  sliding_window(img_binary)
    
    # Assume you now have a new warped binary image 
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = img_binary.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_binary.shape[0]-1, img_binary.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    #get radius
    (left_curverad, right_curverad) = (0,0)#compute_radius(ploty, left_fitx, lefty, right_fitx, righty)        
    return (left_fit, right_fit, left_fitx, right_fitx, left_lane_inds, right_lane_inds, leftx, lefty, rightx, righty)


def sanity_check(img_binary):

    (a,b, aa, bb) = histogram_max(img_binary)
    if a < 50 or b < 50:
        print("$$$$$$$$$$ RESTART1")
        return  False

    if bb - aa < 100:
        print("$$$$$$$$$$ RESTART2 " +str(bb) + " " + str(aa) )
        return False
    
    return True

def confidence(left_radius, right_radius):
    if (left_radius/right_radius) > 99 or (right_radius/left_radius) < 0.01:
        print("$$$$$$$$$$ MISSS")
        return False
    return True    

def drawing(undist, warped, ploty, left_fitx, right_fitx, left_curverad, right_curverad, center, left, right, show=False):
    global Minv

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))    

    #Compute radius
    mean = (left_curverad + right_curverad)/float(2)    
    cv2.putText(newwarp, "Radius: " + str(mean), (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 4, cv2.LINE_AA)

    #Compute center
    cv2.putText(newwarp, "Center OFFSET: " + str(center), (20,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 4, cv2.LINE_AA)

    #Compute left
    cv2.putText(newwarp, "Left Lane: " + str(left), (20,90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 4, cv2.LINE_AA)

    #Compute right    
    cv2.putText(newwarp, "Right Lane: " + str(right), (20,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 4, cv2.LINE_AA)
    
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)    
    if show:
        plt.imshow(result)
        plt.show()
    return result

#Sum up
left_fit=None
right_fit=None
def process_image(img):
    global left_fit, right_fit
    undistorted = undistort_image(img)
    warped = warp(undistorted)
    gradient = combined_gradient(warped)
    new = False
    if left_fit is None or right_fit is None or not sanity_check(gradient) or LEFT_LINE.miss > 5:
        new = True
        (left_fit, right_fit, left_fitx, right_fitx, left_lane_inds, right_lane_inds, leftx, lefty, rightx, righty) = sliding_window(gradient)
    else:
        (left_fit, right_fit, left_fitx, right_fitx, left_lane_inds, right_lane_inds, leftx, lefty, rightx, righty) = fit_again(gradient, left_fit, right_fit)

    if not(left_fit is None or right_fit is None):
        (left_curverad, right_curverad, center, left_center, right_center) = compute_metrics(LEFT_LINE.ploty, left_fitx, right_fitx, gradient.shape[0])
        if confidence(left_curverad, right_curverad):
            if new:
                LEFT_LINE.reset(gradient.shape[1], gradient.shape[0])
                RIGHT_LINE.reset(gradient.shape[1], gradient.shape[0])
            LEFT_LINE.update(left_fit, left_fitx, leftx, lefty, center, left_curverad, left_center, 100, new=True)
            RIGHT_LINE.update(right_fit, right_fitx, rightx, righty, center, right_curverad, right_center, 100, new=True)
        else:
            LEFT_LINE.miss += 1
            RIGHT_LINE.miss += 1
    else:
        LEFT_LINE.miss += 1
        RIGHT_LINE.miss += 1

    return drawing(undistorted, gradient, LEFT_LINE.ploty, LEFT_LINE.bestx, RIGHT_LINE.bestx, LEFT_LINE.radius_of_curvature, RIGHT_LINE.radius_of_curvature, LEFT_LINE.line_base_center_pos, LEFT_LINE.line_base_pos, RIGHT_LINE.line_base_pos)

def pipeline():
    white_output = 'tack.mp4'
    clip1 = VideoFileClip("project_video.mp4")
    white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)



##EXECUTION

pipeline()

#calibrate_camera('camera_cal')
print("Teste Calibration")
#img = mpimg.imread('test_images/straight_lines2.jpg')
img = mpimg.imread('test_images/test4.jpg')
undistorted = undistort_image(img)
region = undistorted #region_of_interest(undistorted)
calibrate_warp_lane(region, show_points=True)
warp = warp(undistorted)
#show_images(undistorted, warp)
#gradient = test_gradient(warp, ksize=15, thresh=(20, 100), direction_thresh=(0.7, 1.3))
gradient = combined_gradient(warp)
#show_images(warp, gradient)
#histogra_gradient(gradient)

(left_fit, right_fit, left_fitx, right_fitx, left_lane_inds, right_lane_inds, left_curverad, right_curverad) = sliding_window(gradient)
plot_curve(gradient, left_fitx, right_fitx, left_lane_inds=left_lane_inds, right_lane_inds=right_lane_inds)
drawing(undistorted, gradient, ploty, left_fitx, right_fitx)
print(left_curverad)
print(right_curverad)
print((left_curverad + right_curverad)/2) 
#(ploty, left_fit, right_fit, left_fitx, right_fitx, left_lane_inds, right_lane_inds, left_curverad, right_curverad) = fit_again(gradient, left_fit, right_fit)
#plot_curve(gradient, left_fitx, right_fitx, left_lane_inds=left_lane_inds, right_lane_inds=right_lane_inds)


