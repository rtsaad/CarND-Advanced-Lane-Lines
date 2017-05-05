import os
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

#Project imports
import camera
import lane
import gradient

LEFT_LINE  = lane.Line(1280, 720)
RIGHT_LINE = lane.Line(1280, 720)

#Sanity Check
def sanity_check(img_binary):

    (l,r, ll, rr) = lane.histogram_max(img_binary)
    if l < 40 or r < 40:        
        return  False

    if abs(rr - ll) < 30:
        return False
    
    return True

def confidence(left_radius, right_radius):
    if (left_radius/right_radius) > 99 or (right_radius/left_radius) < 0.01:
        return 0
    if left_radius > right_radius:
        return float(1)/(float(left_radius)/right_radius)
    return float(1)/(float(right_radius)/left_radius)

#Sum up
left_fit=None
right_fit=None
def process_image(img):
    global left_fit, right_fit
    img_undistorted = camera.undistort_image(img)
    img_warped = camera.warp(img_undistorted)
    img_gradient = gradient.combined_gradient(img_warped)
    
    if left_fit is None or right_fit is None or not sanity_check(img_gradient) or LEFT_LINE.miss > 10:
        LEFT_LINE.miss += 1               
        (left_fit, right_fit, left_fitx, right_fitx, left_lane_inds, right_lane_inds, leftx, lefty, rightx, righty) = lane.sliding_window(img_gradient)
    else:
        (left_fit, right_fit, left_fitx, right_fitx, left_lane_inds, right_lane_inds, leftx, lefty, rightx, righty) = lane.fit_again(img_gradient, LEFT_LINE.best_fit, RIGHT_LINE.best_fit)#left_fit, right_fit)
        

    if not(left_fit is None or right_fit is None):
        (left_curverad, right_curverad, center, left_center, right_center) = lane.compute_metrics(LEFT_LINE.ploty, left_fitx, right_fitx, img_gradient.shape[1])
        conf = confidence(left_curverad, right_curverad)
        if conf!=0:
            if LEFT_LINE.miss > 10:                
                LEFT_LINE.reset(img_gradient.shape[1], img_gradient.shape[0])
                RIGHT_LINE.reset(img_gradient.shape[1], img_gradient.shape[0])
            LEFT_LINE.update(left_fit, left_fitx, leftx, lefty, center, left_curverad, left_center, conf)
            RIGHT_LINE.update(right_fit, right_fitx, rightx, righty, center, right_curverad, right_center, conf)
            if conf > 0.5:
                LEFT_LINE.miss -= 1
                RIGHT_LINE.miss -= 1
        else:
            LEFT_LINE.miss += 1
            RIGHT_LINE.miss += 1
    else:
        LEFT_LINE.miss += 1
        RIGHT_LINE.miss += 1

    #if conf > 0.99:
    #    return camera.drawing(img_undistorted, img_gradient, LEFT_LINE.ploty, left_fitx, right_fitx, left_curverad, right_curverad, center, left_center, right_center)
    #else:
    return camera.drawing(img_undistorted, img_gradient, LEFT_LINE.ploty, LEFT_LINE.bestx, RIGHT_LINE.bestx, LEFT_LINE.radius_of_curvature, RIGHT_LINE.radius_of_curvature, LEFT_LINE.line_base_center_pos, LEFT_LINE.line_base_pos, RIGHT_LINE.line_base_pos)


#Flags
import argparse
parser = argparse.ArgumentParser(description='Advanced Lane Detection.')
parser.add_argument('video_input', type=str, metavar='project_video', help="Video to process")
parser.add_argument('video_output', type=str, metavar='track', help="file output")

#Resume pipeline
def pipeline():
    args = parser.parse_args()
    white_output = 'output_images/' + args.video_output + '.mp4'
    clip1 = VideoFileClip(args.video_input + ".mp4")
    white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)

##Execute
pipeline()
