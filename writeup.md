**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./lanelineimgs/undistort_output.png "Undistorted"
[image2]: ./lanelineimgs/testundistort.png "Road Transformed"
[image3]: ./lanelineimgs/binary.png "Binary Example"
[image4]: ./lanelineimgs/straightlinepersp.png "Warp Example"
[image5]: ./lanelineimgs/lanefit.png "Fit Visual"
<<<<<<< HEAD
[image6]: ./lanelineimgs/finaloutput.jpg "Output"
[video1]: ./project_video.avi "Video"
=======
[image6]: ./lanelineimgs/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"
>>>>>>> origin/master

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in lines #12 through #31 (obj points) and line #264 of the file called `hoganlanefinder.py`  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function in line #264.  I applied this distortion correction to the test image using the `cv2.undistort()` function in line #55 and obtained this result: 

![alt text][image1]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will show how I applied the distortion correction to one of the lane images like this one:
![alt text][image2]
####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color thresholds to generate a binary image (thresholding steps at lines #42 through #50 in `another_file.py`).  Here's an example of my output for this step.

![alt text][image3]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp_image()`, which appears in lines #34 through #39 in the file `hoganlanefinder.py` The `warp_image()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points along with image size.  I chose the hardcode the source and destination points in my 'perspective_shift()' function like so:

```
src = np.float32(##defining source points
        [
            (780, 500),  # top right
            (1140, 700),  # bottom right
            (204, 700),  # bottom left
            (516, 500)  # top left
        ]
    )
offset = 250
dst = np.float32(##defining destination points
        [
            (img_size[0]-offset, 0),  # top right
            (img_size[0]-offset, img_size[1]),  # bottom right
            (0+offset, img_size[1]),  # bottom left
            (0+offset, 0)  # top left
        ]
    )

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 780, 500      | 1030, 0       | 
| 1140, 700     | 1030, 1280    |
| 204, 700      | 250, 1280     |
| 516, 500      | 250, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image - which as you can see from the perspective shift performed on the straight lane line example, they do.

![alt text][image4]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines #87 through #129 in my code in `hoganlanefinder.py` Both polynomial fits taken from the birds eye view was used to derive the radius of curvature for each lane line given a defined pixel to meter ratio. By defining it as 30/720 y meters per pixel and 3.7/700 x meters per pixel, I was able to keep the lane curvature radius within an order of magnitude of the expected curvature radius. It also never went below the expected radius, which is where the real danger would be when a self driving car may unexpectedly curve too tightly. The radius increases towards infinity for straight sections of the road.

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines #230 through #250 in my code in `hoganlanefinder.py` in the function `drawlines()`.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./hoganlanefinding.avi)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The code performed well in defining  the lane area and plotting it onto video. However, the current structure predicts the radius and distance from the center of the predicted lane lines found only using the current frame. This is an artifact of how I started the project to predict lane lines given an image and not a video. As a result, the curvature radius and distance from center is much noisier than the final output - which weights the lane line predictions with the previous lane line predictions on every frame except the first one. If a self driving vehicle were to utilize that visually predicted information in any way (which I think is a poor choice) then it would not perform well. This is a simple fix though, as all that needs to be done is perform another call to the curvature prediction function after the weighted lane line has been predicted. 

Another issue is that the code is very fitted to the current project. Import names will need to be changed for any other applications, source points will need to be changed if the camera is shifted at all, and it would likely fail in attempting to switch lanes. Creating conditional statements for rare lane cases - such as when the lane lines found with the traditional method fails sanity checks - could be used to teach the program how to identify when a vehicle is in between lanes and which direction it is heading.

