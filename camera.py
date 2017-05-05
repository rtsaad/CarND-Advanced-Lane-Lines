import os
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


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

def show_images(origin, undistorted, title1="", title2="", filename=None):
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
    if filename is not None:
        plt.savefig('output_images/'+filename+'.png', dpi=100)
    plt.show()

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
                plt.savefig('output_images/corners.png', dpi=100)
                plt.show()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print("Calibration Finished")
    print("Save parameters")
    output = open(CAMERA_PARAMETERS_FILE, 'wb')    
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
        [556, 476], 
        [730, 476], 
        [1060, 690],
        [259, 690]  
    ])
    # Destination points 
    offset_X = 1058
    offset_Y = 704
    dst = np.float32([
        [259, 200],
        [1060, 200],
        [1060, 720],
        [259, 720]
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
        plt.savefig('output_images/warp_points.png', dpi=100)
        plt.show()
        warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
        plt.imshow(warped)
        plt.plot(dst[0][0], dst[0][1], '.')
        plt.plot(dst[1][0], dst[1][1], '.')
        plt.plot(dst[2][0], dst[2][1], '.')
        plt.plot(dst[3][0], dst[3][1], '.')
        plt.savefig('output_images/warp_perspective.png', dpi=100)
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

    #Print radius
    mean = (left_curverad + right_curverad)/float(2)    
    cv2.putText(newwarp, "Radius: " + ("{0:.3f} (meters)".format(mean)), (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 4, cv2.LINE_AA)

    #Print center
    cv2.putText(newwarp, "Center OFFSET: " + ("{0:.3f} (meters)".format(center)), (20,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 4, cv2.LINE_AA)

    #Print left
    #cv2.putText(newwarp, "Left Lane: " + ("{0:.3f} (meters)".format(left)), (20,90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 4, cv2.LINE_AA)

    #Print right    
    #cv2.putText(newwarp, "Right Lane: " + ("{0:.3f} (meters)".format(right)), (20,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 4, cv2.LINE_AA)
    
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)    
    if show:
        plt.imshow(result)
        plt.show()
    return result

if __name__ == "__main__":
    #calibrate_camera('camera_cal', show_corners=False)
    print("Teste Camera")
    img = mpimg.imread('test_images/straight_lines2.jpg')
    #img = mpimg.imread('camera_cal/calibration1.jpg')    
    undistorted = undistort_image(img)
    #show_images(img, undistorted, title1='Orginal', title2='Undistorted', filename='chessboard')    
    calibrate_warp_lane(undistorted, show_points=False)
    #warp = warp(undistorted)
    #show_images(undistorted, warp, title1='Orginal', title2='Warped', filename='warp')
    
