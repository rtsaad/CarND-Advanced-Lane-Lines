# Advanced Lane Finding Project

This project consists of a python script to detect lane lines from an image for self-driving purpose. It uses Image Recognition techniques to identify lane lines and draw it over the original image. The script itself can work over one or a sequence of images (i.e. video).

The main goals of this project is to develop a sequence of operations (pipeline) that gets as input a raw image and outputs an annotate image where the lane lines are marked, the lane curvature and . Figures 1 and 2 depict examples of the input and output images.

![alt text][image1]

[//]: # (Image References)

[image1]: output_images/output.png "Input and Output images"
[image2]: output_images/corners2.jpg "Chessboard Corners"
[image3]: output_images/chessboard.png "Undistorted Chessboard"
[image4]: output_images/gradient_undistorted.png "Gradient"
[image5]: output_images/warp.png "Bird-Eye View"
[image6]: output_images/corners_window.png "Lane Line Detection" 
[image7]: output_images/lane.png "Lane Line Plot" 


## 1.Access 

The source code for this project is available at [project code](https://github.com/otomata/).

## 2.Files

The following files are part of this project:
* camera.py:     Camera functions for calibration and perspective transform;
* gradient.py:   Gradient functions to construc a binary image;
* lane.py:       Lane functions to keep track, smooth and plot the lane into the video output;
* pipeline.py:   Script to execute the lane detction pipeline of operations;
* output_images: 
** chessboard.png:  cherssboard before and after undistor operation;
** corners*.png:    OpenCv chessboard corners detected;
** corners_window.png: polinomial fit over the lane lines;
** gradient.png:    gradient operation over warped image;
** gradient_undistorted: gradient operation over undistorted image;
** track1.mp4: lane detection for project_video.mp4;
** track2.mp4: lane detection for challenge_video.mp4;
** track3.mp4: lane detection for harder_challenge_video.mp4;
** undistort.png: undistort operation over original image;
** warp.png: warp tranformation (bird-eye view) over original image;
** warp_points.png: original image before warp tranformation with source points;
** warp_perspective.png: warped image with destination points.


### Dependency

This project requires the following python packages to work:
* openvc
* numpy
* matplotlib
* moviepy
* pickle

## 3.How to use this project

### Executing the Pipeline

To detect the lane lines, call the following command, together with the file names for the input and output videos.

```sh

python pipeline.py input_video output_video

```

## 4. Project - Introduction

The goals of this project is to detect Lane Lines from a video streaming. Therefore, there are fives necessary steps to accomplish this task, they are:

* Camera calibration to remove distortion from the camera images;
* Color manipulation and gradient transforms to construct a binary image;
* Perspective transform ("birds-eye view") to improve the binary image;
* Detect the lane lines;
* Determine the curvature of the lane and vehicle position with respect to center;
* Plot the detected lane boundaries back onto the original image;
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## 5. Camera Calibration

The first step in this project is to undo the distortion caused by the camera when 3d objects are transformed into a 2d image. This is a mandatory step because the distortion can change the apparent size and shape of an object in an image.

For this step, we use helper functions from the OpenCv package to compute distortion coefficients and the camera matrix. This calibration process requires 20 images of a chessboard calibrate the camera. The first step is to map "image points", which are the corners detected from the images, to the "object points", which are (x,y,z) coordinates of the chessboard corners in the real world. The `object points` are in fact the same for all the images and the `image points` are detected using the OpenCv function `findChessboardCorners`. Finally, we use the OpenCv helper function `calibrateCamera` to calculate the distortion coefficients and the camera matrix. Figures 2 and 3 show the corner detection and the undistorted image of the chessboard.

![alt text][image2]

![alt text][image3]

The code for this step is contained at function `calibrate_camera` from  the `camera.py` file (lines 46-80).

## 6. Color Transformation (Gradient)

The second step is to construct a binary image with most of the lane lines but without the non-useful parts of the image. We used a combination of gradient and color threshold. For the gradient, we use the `Sobel` operator to take the gradient in the x direction to emphasizes edges closer to vertical. We also use the hls and hsv color spaces to build a binary image to be reliable as possible to different lighting conditions. For the hls color space, we threshold the saturation channel because it is less affected by the lighting conditions. In addition, we also use the hsv color space to isolate the colors yellow and white to enhance the highlight of the lane lines. The code below depicts the combination of the gradient, the hls and hsv threshold. Figure 4 shows the obtained binary image.

"""
combined[((hue == 1) & (hls == 1)) | (gradx == 1) ] = 1

"""

![alt text][image4]

The code for this step is contained at function `combined_gradient` from  the `gradient.py` file (lines 101-108).

## 7. Perspective transform ("birds-eye view") 

The birds-eye view allows to view the lane from above, helping the calculate the lane curvature later. 

We use the OpenCv `getPerspectiveTransform` to compute the perspective transform `M` which maps points (rectangle) from in a given image (source) to a different desired image points with a new perspective (destination). The procedure to define the source and destination rectangles (points) followed an empirical process of trial and errors until the following source and destination points were obtained:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 556,  476     | 259, 200      | 
| 730,  476     | 1060, 200     |
| 1060, 690     | 1060, 720     |
| 259,  690     | 259, 720      |

Figure 5 shows the bird-eye view of the image using the source and destinations points presented above.

![alt text][image5]

The code for this step is contained at function `calibrate_warp_lane` from  the `camera.py` file (lines 128-172).

## 8. Lane Line Fit

The lane line detection starts with the analysis of the image histogram to identify the base of the right and left lines. Then, we use sliding windows from bottom-up to identify non zero pixels for both left and right sides. After we have identified all nonzero pixels, we fit these pixels into a 2nd order polynomial using the numpy package function `polyfit`. The code for this step is contained at function `sliding_window` from  the `lane.py` file (lines 149-251). Figure 6 presents the identified lane-line pixels with the fitted polynomial.

![alt text][image6]

Once we have good polynomial fit, we skip the sliding windows and search on the next frame using the information from the previous one. The code for this step is contained at function `fit_again` from  the `lane.py` file (lines 253-285). In addition, we also have a `Line` class to keep track of the last 45 lane lines (or frames) we have detected. We use a confidence value to compute a weighted average in order to smooth the lane lines and to obtain a cleaner result. Our confidence value analyzes how parallel are the left and right lines, given a value (weight) of 1 when parallel or 0 when nonparallel. Below, we present our code to calculate the confidence value:

"""
def confidence(left_radius, right_radius):
    if (left_radius/right_radius) > 99 or (right_radius/left_radius) < 0.01: 
        return 0
    if left_radius > right_radius:
        return float(1)/(float(left_radius)/right_radius)
    return float(1)/(float(right_radius)/left_radius)

"""

The code below present our pipeline to process each image, detect, sanity check and later plot it back onto the road. Our `sanity_check` implementation checks the lanes histograms for a minimum threshold and also for a minimum distance between left and right lanes (file `pipeline.py`, lines 20-29). The code for this step is contained at function `process_image` from  the `pipeline.py` file (lines 44-77). Figure 7 depicts an image of the Lane Line plotted back down onto the road.

"""

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
                print("Reset")
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

    return camera.drawing(img_undistorted, img_gradient, LEFT_LINE.ploty, LEFT_LINE.bestx, RIGHT_LINE.bestx, LEFT_LINE.radius_of_curvature, RIGHT_LINE.radius_of_curvature, LEFT_LINE.line_base_center_pos, LEFT_LINE.line_base_pos, RIGHT_LINE.line_base_pos)

""" 

![alt text][image7]

## 9. Radius Curvature and Center Position

The radius curvature is calculated from the fitted 2nd order polynomial. Given the polynomial f(y)=Ay^2+By+C, we apply the following formula for each lane line to obtain the radius curvature:

"""
Rcurve = ((1 + (2Ay + B)^2)^(3/2))/|2A|
"""

Assuming that the camera is mounted at the center of the vehicle, to compute the position of the vehicle with respect to the center, we first calculate the center of the detected lanes and subtract it from the center of the image. Below, the formula is presented in details:

"""
center_lane = center_lane = (lane_left + (lane_right - lane_left)/2)
center_image = | center_lane - image_widht/2 |
"""

Finaly, we have to convert from pixel to meters. From on our perspective transformation (bird-eye view) source and destination points, we use the following conversions values:

"""
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/800 # meters per pixel in x dimension
"""

The code for this step is contained at function `compute_metrics` from  the `lane.py` file (lines 116-145).

---

### Pipeline (video)

Here is the link for the [project_video output](). From the video, our implementation performs reasonably well on the entire project video, our weighted average smoothed the plotted lanes without any visible catastrophic failures that would cause the car to drive off the road.


---

### Discussion

Our implementation uses techniques (hue color) to isolate the yellow and white colors to maximize line detection. However, our algorithm suffered with the different road colors present on  the [challenge_video](). We were able to partially bypass this problem by using an aggressive weighted average to remove low confidence (lane line) fits (Section 8). Here is the link for the [challenge_video output]().


Another problem we came across is that our implementation was not able to perform reasonably with the sharp turns of the `harder_challenge_vide`. We believe that the main problem is that the 2nd order polynomials are not enough to fit more complex curvatures, such the ones presented in this video. For future work, we consider to try 3nd and 4nd order polynomials to improve these results.
