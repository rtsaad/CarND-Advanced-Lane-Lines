import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#Classes
# Define a class to receive the characteristics of each line detection
class Line():

    n = 45# 45 frames / 1/2 seconds
    def __init__(self, width, height):
        self.reset(width, height)

    def reset(self, width, height):
        #numer of misses
        self.miss = 0
        #const ploty        
        self.ploty = np.linspace(0, height-1, height)
        # was the line detected in the last iteration?
        self.detected = False  #
        # x values of the last n fits of the line
        self.recent_xfitted = []#
        # confidence weights for fitted line
        self.recent_xconfidence = [] #between 0-1
        #average x values of the fitted line over the last n iterations
        self.bestx = None#     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None#
        self.radius_of_curvature_n = []#
        #distance in meters of vehicle center from the lane
        self.line_base_pos = None#
        self.line_base_pos_n = []#
        #distance in meters of vehicle center from the line
        self.line_base_pos_center = None#
        self.line_base_pos_center_n = []#    
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None

    def update(self, fit, fitx, values_x, values_y, center, radius, distance, confidence):
        
        self.detected = True

        if len(self.recent_xfitted) > self.n:            
            self.recent_xfitted.pop(0)
        self.recent_xfitted.append(fitx)

        if len(self.recent_xconfidence) > self.n:            
            self.recent_xconfidence.pop(0)
        self.recent_xconfidence.append(confidence)
        
        if len(self.radius_of_curvature_n) > self.n:
            self.radius_of_curvature_n.pop(0)
        self.radius_of_curvature_n.append(radius)

        if len(self.line_base_pos_n) > self.n:
            self.line_base_pos_n.pop(0)
        self.line_base_pos_n.append(distance)

        if len(self.line_base_pos_center_n) > self.n:
            self.line_base_pos_center_n.pop(0)
        self.line_base_pos_center_n.append(center)        

        self.allx = values_x
        self.ally = values_y
        
        self.current_fit = fit                
        
        #update averages
        self.line_base_pos = np.average(self.line_base_pos_n, weights=self.recent_xconfidence)
        self.line_base_center_pos = np.average(self.line_base_pos_center_n, weights=self.recent_xconfidence)
        self.radius_of_curvature = np.average(self.radius_of_curvature_n, weights=self.recent_xconfidence)
        self.bestx = np.average(self.recent_xfitted, axis=0, weights=self.recent_xconfidence)
        self.best_fit = np.polyfit(self.ploty, self.bestx, 2)        
        #self.bestx = self.best_fit[0]*self.ploty**2 + self.best_fit[1]*self.ploty + self.best_fit[2]

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
    
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/800 # meters per pixel in x dimension
    lx = lx[::-1]  # Reverse to match top-to-bottom in y
    rx = rx[::-1]  # Reverse to match top-to-bottom in y
        
    y_eval = np.max(p)

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
    center = abs((center_lane - center_img)*(xm_per_pix))    

    #Compute left
    left = abs((center_lane - lx[0])*(xm_per_pix))

    #Compute right
    right = abs((rx[0] - center_lane)*(xm_per_pix))
    
    return (left_curverad, right_curverad, center, left, right)

def sliding_window(img_binary, show_image=False):

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
    if show_image:
        out_img = np.dstack((img_binary, img_binary, img_binary))*255
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
        if show_image:
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
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

    if show_image:
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.savefig('output_images/corners_window.png', dpi=100)
        plt.show()

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


if __name__ == "__main__":
    import matplotlib.image as mpimg
    #Project imports
    import camera
    import gradient
    
    print("Test Lane Detection")
    LEFT_LINE  = Line(1280, 720)
    RIGHT_LINE = Line(1280, 720)
    #img = mpimg.imread('test_images/straight_lines2.jpg')
    img = mpimg.imread('test_images/test1.jpg')
    img_undistorted = camera.undistort_image(img)      
    img_warp = camera.warp(img_undistorted)
    img_gradient = gradient.combined_gradient(img_warp)    
    (left_fit, right_fit, left_fitx, right_fitx, left_lane_inds, right_lane_inds, leftx, lefty, rightx, righty) = sliding_window(img_gradient)
    (left_curverad, right_curverad, center, left_center, right_center) = compute_metrics(LEFT_LINE.ploty, left_fitx, right_fitx, img_gradient.shape[1])
    img_plot = camera.drawing(img_undistorted, img_gradient, LEFT_LINE.ploty, left_fitx, right_fitx, left_curverad, right_curverad, center, left_center, right_center)
    camera.show_images(img_warp, img_plot, title1='Warped', title2='Lanes', filename='lane')
