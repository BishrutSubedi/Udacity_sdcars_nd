import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

# Define our parameters for Canny and apply
low_threshold = 50
high_threshold = 150
kernel_size = 5

# Define the Hough transform parameters
# Make a blank the same size as our image to draw on
rho = 2 # distance resolution in pixels of the Hough grid
theta = np.pi/180 # angular resolution in radians of the Hough grid
threshold = 25     # minimum number of votes (intersections in Hough grid cell)
min_line_length = 10 #minimum number of pixels making up a line
max_line_gap = 10    # maximum gap in pixels between connectable line segments
# line_image = np.copy(image)*0 # creating a blank to draw lines on

alpha = 0.95
filtered_pos_slope = 0
filtered_pos_intercept = 0
filtered_neg_slope = 0
filtered_neg_intercept =  0
count = 0

# Create a VideoCapture object and read from input file 
cap = cv2.VideoCapture('test_videos/solidWhiteRight.mp4') 
# cap = cv2.VideoCapture('test_videos/solidYellowLeft.mp4') 

#challenge not yet working
# cap = cv2.VideoCapture('test_videos/challenge.mp4')

length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('Output_video/test_output_syl.avi',fourcc, 20.0, (frame_width,frame_height))
out = cv2.VideoWriter('Output_video/test_output_swr.avi',fourcc, 20.0, (frame_width,frame_height))

print("\nVideo Processing started...\n") 

# Read until video is completed 
while(cap.isOpened()):
   
  # Capture frame-by-frame 
  ret, frame = cap.read() 
  if ret == True:
        count = count + 1
        # Redefining frame to grayscale
        # frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
        grey_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        line_image = np.copy(frame)*0 # creating a blank to draw lines on

        grey_blur = cv2.GaussianBlur(grey_image,(kernel_size, kernel_size),0)

        edges = cv2.Canny(grey_blur, low_threshold, high_threshold)

        # Next we'll create a masked edges image using cv2.fillPoly()
        mask = np.zeros_like(edges)   
        ignore_mask_color = 255   

        # This time we are defining a four sided polygon to mask
        imshape = frame.shape #prev was image.shape
        vertices = np.array([[(900,539),(65, 539), (450, 325), (490,325)]], dtype=np.int32)
        # vertices = np.array([[(850,539),(120, 539), (450, 350), (490, 350)]], dtype=np.int32)
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        masked_edges = cv2.bitwise_and(edges, mask)

        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)
        
        #declare empty array to store values of slope, intercept and line points
        slope_array_pos = np.array([])
        slope_array_neg = np.array([])

        intercept_arr_pos = np.array([])
        intercept_arr_neg = np.array([])

        line2_array = np.empty((0,4))
        line1_array = np.empty((0,4))

    # Iterate over the output "lines" and draw lines on a blank image
        for line in lines:
            for x1,y1,x2,y2 in line:
                a = (x2 - x1)
                b = (y2 - y1)
                
                if(a != 0): # and b !=0):
                    slope = b/a
                    intercept = -slope *x1 + y1
                #find intercept as well

                if (slope > 0):
                    slope_array_pos = np.append(slope_array_pos, slope)
                    intercept_arr_pos = np.append(intercept_arr_pos,intercept)
                    line1_array = np.concatenate((line1_array, line))
                    #cv2.line(line_image,(line1[0,0],line1[0,1]),(line1[0,2],line1[0,3]),(255,0,0),10)
                else:
                    slope_array_neg = np.append(slope_array_neg, slope)
                    intercept_arr_neg = np.append(intercept_arr_neg,intercept)
                    line2_array = np.concatenate((line2_array, line)) 
                    #cv2.line(line_image,(line2[0,0],line2[0,1]),(line2[0,2],line2[0,3]),(255,0,0),10)
                    #cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
        
            # Create a "color" binary image toimshow combine with line image
            color_edges = np.dstack((edges, edges, edges)) 
            # Draw the lines on the edge image
            lines_edges = cv2.addWeighted(frame, 0.8, line_image, 1, 0)    #prev(image,0.8,...)

        # print(len(slope_array_pos))

        if ( (len(slope_array_pos) != 0) and (len(slope_array_neg) != 0 )):
            average_pos_slope = sum(slope_array_pos)/len(slope_array_pos)
            average_intercept_pos = sum(intercept_arr_pos)/len(intercept_arr_pos)
            average_neg_slope = sum(slope_array_neg)/len(slope_array_neg)
            average_intercept_neg = sum(intercept_arr_neg)/len(intercept_arr_neg)
        
        if (count == 1):
            filtered_pos_slope = average_pos_slope
            filtered_pos_intercept = average_intercept_pos
            filtered_neg_slope = average_neg_slope
            filtered_neg_intercept =  average_intercept_neg
        else:
            filtered_pos_slope = alpha * filtered_pos_slope + average_pos_slope * (1-alpha)
            filtered_pos_intercept = alpha * filtered_pos_intercept + average_intercept_pos * (1-alpha)
            filtered_neg_slope = alpha * filtered_neg_slope + average_neg_slope * (1-alpha)
            filtered_neg_intercept =  alpha * filtered_neg_intercept + average_intercept_neg * (1-alpha)

    #give y and find x using m and c {do ittwice for two x and y}

        y_2 = 540 #y1 has to be the max value in the array
        y_1 = 330
        x_p1 = int((y_1 - filtered_pos_intercept) / filtered_pos_slope)
        x_p2 = int((y_2 - filtered_pos_intercept) / filtered_pos_slope)

        x_n1 = int((y_1 - filtered_neg_intercept) / filtered_neg_slope)
        x_n2 = int((y_2 - filtered_neg_intercept) / filtered_neg_slope)

        cv2.line(line_image,(x_p1,y_1),(x_p2,y_2),(0,0,255),12)

        cv2.line(line_image,(x_n1,y_1),(x_n2,y_2),(0,0,255),12)

        final_frame = cv2.bitwise_or(line_image, frame)
        out.write(final_frame)
   
    # Press Q on keyboard to  exit 
        if cv2.waitKey(25) & 0xFF == ord('q'): 
            break 
  # Break the loop 
  else:
      break  
    # break
   
# When everything done, release the video capture object 
# This statement releases the webcam but we don't have webcam or ?
print("Video Processing complete.")
cap.release() 
out.release()
   
# Closes all the frames 
cv2.destroyAllWindows() 
