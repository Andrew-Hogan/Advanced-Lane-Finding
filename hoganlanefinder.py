import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
##The main function is near the bottom. Image plotting/showing used during testing has been commented out for batch rendering purposes (imagine closing 616 x 6 popups by hand?)


##This function finds the chessboard corners used in undistorting images taken by this camera lens within the perspective_shift function
def findcal(num_samples):
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)
    
    objpoints = []
    imgpoints = []
    counter = 1
    while counter < (num_samples+1):
        img = cv2.imread('./camera_cal/calibration' + str(int(counter)) + '.jpg')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        # If found, draw corners
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
        counter += 1
    return objpoints, imgpoints

##This function returns a birds eye view of the input image given source and destination points
def warp_image(img,src,dst,img_size):
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    Minv = cv2.getPerspectiveTransform(dst, src)

    return warped,M,Minv

##This function isolates lane lines by enforcing multiple color threshholds and turning off all pixels which do not meet the criteria
def colorthresh(img):
    R = img[:,:,0]
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    S = hls[:,:,2]
    thresh = (90, 255)
    binary = np.zeros_like(S)
    binary[(S > thresh[0]) & (S <= thresh[1]) & (R > thresh[0]) & (R <= thresh[1]) ] = 1
    #cv2.imshow("test color", binary)
    return binary

##This is the function which takes an image along with chessboard corners for a given lens and returns a birds eye binary view of the lane lines - it references colorthresh and warp_image
def perspective_shift(img, mtx, dist):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    undistort = cv2.undistort(img, mtx, dist, None, mtx) ##here is where the undistorted image is produced
    img_size = (undistort.shape[1], undistort.shape[0]) 
    
    src = np.float32(##defining source points
        [
            (780, 500),  # top right
            (1140, 700),  # bottom right
            (204, 700),  # bottom left
            (516, 500)  # top left
        ]
    )
    offset = 250
    #plt.imshow(undistort)
    #plt.plot(780, 500, '.')
    #plt.plot(1140, 700, '.')
    #plt.plot(204, 700, '.')
    #plt.plot(516, 500, '.')
    #plt.show()
    dst = np.float32(##defining destination points
        [
            (img_size[0]-offset, 0),  # top right
            (img_size[0]-offset, img_size[1]),  # bottom right
            (0+offset, img_size[1]),  # bottom left
            (0+offset, 0)  # top left
        ]
    )
    threshed = colorthresh(img)##produces binary image
    warped,M,Minv_warp = warp_image(threshed,src,dst,(img_size[0],img_size[1]))##birds eye view
    
    return warped, M, Minv_warp, undistort

##finds the curve radius for a lane line image
def findcurve(left_fit, right_fit):
    ploty = np.linspace(0, 719, num=720)# to cover same y-range as image
    # For each y position generate random x position within +/-50 pix
    # of the line base position in each case (x=200 for left, and x=900 for right)
    leftx = np.array([left_fit[2] + (y**2)*left_fit[0] + (y)*left_fit[1] + np.random.randint(-50, high=51) 
                              for y in ploty])
    rightx = np.array([right_fit[2] + (y**2)*right_fit[0] + (y)*right_fit[1] + np.random.randint(-50, high=51) 
                              for y in ploty])

    # Fit a second order polynomial to pixel positions in each lane line
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Plot up the data
    #mark_size = 3
    #plt.plot(leftx, ploty, 'o', color='red', markersize=mark_size)
    #plt.plot(rightx, ploty, 'o', color='blue', markersize=mark_size)
    #plt.xlim(0, 1280)
    #plt.ylim(0, 720)
    #plt.plot(left_fitx, ploty, color='green', linewidth=3)
    #plt.plot(right_fitx, ploty, color='green', linewidth=3)
    #plt.gca().invert_yaxis() # to visualize as we do the images
    #plt.show()
    # Define y-value where we want radius of curvature
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    print(left_curverad, right_curverad) ##output the radius

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Output the radius in meters
    print(left_curverad, 'm', right_curverad, 'm')

    return left_fitx, right_fitx, left_curverad, right_curverad

#This function finds where the lane lines start at the bottom of a binary birds-eye image, calculates a polynomial regression for that lane line, and then references findcurve to find the curve radius
def findstart(binary_warped):
    histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0) #histogram of most frequent pixel locations in binary image
    # Create an output image to draw on and  visualize the result - the actual showing part of this code has been commented out for batch rendering purposes
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    #find car's location relative to center line through averaging histogram density locations versus expected lane line locations given perfect center driving
    shift = leftx_base - 250
    shift2 = rightx_base - (histogram.shape[0] - 250)
    print (shift)
    print (shift2)
    xm_per_pix = 3.7/700
    lanedif = abs(shift - shift2)
        

    shift = (shift + shift2)/2
    shift *= xm_per_pix
    print (shift, 'm right of center') #output the calculated distance; negative meaning left of center

    #here the function uses the sliding windows technique to isolate which pixels belong to which lane
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
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
    
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
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

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    #ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    #left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    #right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    #plt.imshow(out_img)
    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='yellow')
    #plt.xlim(0, 1280)
    #plt.ylim(720, 0)
    #plt.show()

    leftx, rightx, cl, cr = findcurve(left_fit, right_fit) ##finds the curve radius

    return leftx, rightx, lanedif, shift, cl, cr

# This function draws the lane lines on the final output image
def drawlines(left_fitx, right_fitx, img, Minv, image, undist):
    ploty = np.linspace(0, 719, num=720)
    # Create an image to draw the lines on
    img_zero = np.zeros_like(img).astype(np.uint8)
    color_img = np.dstack((img_zero, img_zero, img_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_img, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_img, Minv, (image.shape[1], image.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    #plt.imshow(result)
    #plt.show()
    return result


##Start of main
objpoints, imgpoints = findcal(20) ##calibration points

firstrun = True #Used to define whether to incorporate previous lane line predictions
vid = cv2.VideoCapture('project_video.mp4') #Video file to be analyzed
counter = 0 #Output batch number

while(True): #vid.read returns False at the end of a sample video
    #img = cv2.imread('test_images/test6.jpg') #For single images instead of video
    ret, img = vid.read()
    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None) ##Find camera calibration coefficients

    warped, persp, Minv_warp, undist = perspective_shift(img, mtx, dist) ##Creates a binary birds eye view of the image
    #cv2.imshow("test", warped)

    tolerance = 100 #How much of a difference in predicted lane line relative positioning versus expected relative positioning is allowed
    newleft, newright, dif, pos, cl, cr = findstart(warped) #Finds the lane lines and polynomial fit
    if firstrun == True: ##If there is no previous data then we take the lane line prediction regardless
        laneleft = newleft
        laneright = newright
        firstrun = False
    elif dif < tolerance: ##If it is not within tolerance of expected values then we use only the previous prediction
        laneleft = np.sum([(laneleft * .9),(newleft * .1)], axis=0)
        laneright = np.sum([(laneright * .9), (newright * .1)], axis=0)


    final_img = drawlines(laneleft, laneright, warped,Minv_warp, img, undist) ##Draw the lane & lines on the image
    final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB) #Change format for image saving
    cv2.imwrite('./video_render_nonumbers/video_' + str(int(counter)) + '.jpg', final_img)
    current_img = Image.open('./video_render_nonumbers/video_' + str(int(counter)) + '.jpg')#take lane image and superimpose the lane radius curvature for the predicted lanes and distance from center
    draw = ImageDraw.Draw(current_img)
    font = ImageFont.load_default()
    draw.text((0,0),"Left Curve Radius: " + str(cl) + " m;   Vehicle Distance from center: " + str(pos) + " m;   Right Curve Radius: " + str(cr) + " m", (255,255,255), font = font)
    current_img.save('./video_render/video_' + str(int(counter)) + '.jpg')
    counter += 1 #For next batch sample
