import cv2 as cv
import numpy as np
import math
from operator import itemgetter
#from matplotlib import pyplot as plt
#import sys




MIN_MATCH_COUNT = 10
color = np.random.randint(0, 255, (500, 3))
PreviousPoints = []
PreviousPoints2 = []
PreviousPoints3 = []
PreviousPoints4 = []

setPoints = []
setPoint = [0, 0, 0, 0]
setPoints.append(setPoint)

goalLengths = []
goalPostLength = 1
goalLengths.append(goalPostLength)

#CHECK PATH!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!CHECK PATH
videoCapture = cv.VideoCapture('InputVideoCOMP.mp4')
if (videoCapture.isOpened()== False):
    print("Error opening video file")

'''
#Making the video capturer
fps = int(round(videoCapture.get(5)))
frame_width = int(videoCapture.get(3))
frame_height = int(videoCapture.get(4))
fourcc = cv.VideoWriter_fourcc(*'mp4v') 
output_video_file = 'output.mp4'
output = cv.VideoWriter(output_video_file, fourcc, fps, (frame_width, frame_height))
'''

# Function to detect edges using Canny edge detection
def detect_edges(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) # Convert image to grayscale
    blurred = cv.GaussianBlur(gray, (5, 5), 1) # Apply Gaussian blur to reduce noise
    edges = cv.Canny(blurred, 30, 100) # Canny edge detection
    return edges, gray

# Function to detect lines using Hough Line Transform
def detect_hough_lines(edges):
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=200, minLineLength=150, maxLineGap=20)
    return lines

# Function for color segmentation (only green color)
def color_segmentation_green(image):
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([50, 255, 255])
    mask_green = cv.inRange(hsv_image, lower_green, upper_green)
    segmented_image = cv.bitwise_and(image, image, mask=mask_green)
    return segmented_image

#Function for applying a saturation mask to a frame
def increase_image_saturation(image, saturation = 2):
    imghsv = cv.cvtColor(image, cv.COLOR_BGR2HSV).astype("float32")
    (h, s, v) = cv.split(imghsv)
    s = s*saturation
    s = np.clip(s,0,255)
    imghsv = cv.merge([h,s,v])
    saturatedImage = cv.cvtColor(imghsv.astype("uint8"), cv.COLOR_HSV2BGR)
    return saturatedImage

#Function for applying a contrast and brightness mask to a frame
def change_image_brightness_contrast(image, brightness, contrast):
    editedImage = cv.addWeighted(image, contrast, np.zeros(TestImage.shape, TestImage.dtype), 0, brightness) 
    return editedImage

#Function for applying a deletion mask to a color segmented image
def delete_outside_area(image):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (6, 3))
    TestMask = cv.erode(image, kernel, iterations=15)
    TestMask = cv.cvtColor(TestMask, cv.COLOR_BGR2GRAY)
    empty, TestMask = cv.threshold(TestMask, 0, 255, cv.THRESH_BINARY)
    TestMask = cv.dilate(TestMask, kernel, iterations=30)
    TestMask = cv.cvtColor(TestMask, cv.COLOR_GRAY2BGR)
    filteredimage = cv.bitwise_and(image, TestMask)
    return filteredimage

# Function to calculate the length of a line
def calculate_line_length(line):
    x1, y1, x2, y2 = line[0]
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Function to calculate the angle of a line
def calculate_line_angle(line):
    x1, y1, x2, y2 = line #was [0]
    angle_radians = np.arctan2(y2 - y1, x2 - x1)
    angle_degrees = np.degrees(angle_radians)
    return angle_degrees

# Function to find and draw parallel and perpendicular lines with angle thresholds
def find_and_draw_parallel_perpendicular(image, lines, longest_line):
    if longest_line is None:
        return None

    longest_angle = calculate_line_angle(longest_line)
    parallel_threshold_low = 0
    parallel_threshold_high = 6
    perpendicular_threshold_low = 70
    perpendicular_threshold_high = 80
    parallel_length_threshold = 250
    perpendicular_length_threshold = 150

    combined_image = image.copy()

    for line in lines:
        angle = calculate_line_angle(line)
        length = calculate_line_length(line)
        angle_relative = abs(angle - longest_angle) % 180

        if (parallel_threshold_low <= angle_relative <= parallel_threshold_high) and length > parallel_length_threshold:
            cv.line(combined_image, tuple(line[0][:2]), tuple(line[0][2:]), (0, 255, 0), 2)

        if (perpendicular_threshold_low <= abs(angle_relative - 90) <= perpendicular_threshold_high) and length > perpendicular_length_threshold:
            cv.line(combined_image, tuple(line[0][:2]), tuple(line[0][2:]), (0, 0, 255), 2)

    return combined_image

def getMaxLines(edges):

        # Step 3: Morphological operations (JANNAT) customised
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
        dilated_edges = cv.dilate(edges, kernel, iterations=1)
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (6, 6))
        eroded_edges = cv.erode(dilated_edges, kernel, iterations=1)
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
        eroded_edges = cv.dilate(eroded_edges, kernel, iterations=1)

        # Step 4: Detect lines using Hough Line Transform (JANNAT)
        lines = detect_hough_lines(eroded_edges)
        hough_image = TestImage.copy()

        if lines is not None:
            if lines.all():
                for i in range(0, len(lines)):
                    l = lines[i][0]
                    cv.line(hough_image, (l[0], l[1]), (l[2], l[3]), color[i].tolist(), 3, cv.LINE_AA)

        #Values for filtering and grouping
        minLength = 10
        minRes = 1
        maxAngleDifference = 5
        maxDistanceDifference = 15

        #Filtering and grouping lines
        skipIndex = []
        LinesSorted = []
        if lines is not None:

            #Load first line
            for i, line1 in enumerate(lines):
                skipLoop = False
                line1 = line1[0]
                v1 = line1[2:4] - line1[0:2]
                len1 = np.linalg.norm(v1)
            
                for s in range(0, len(skipIndex)):
                    skip = skipIndex[s]
                    if skip == i:
                        skipLoop = True
                        break
                
                #Skip it its too short or if it has allready been appended      
                if len1 < minRes * minLength or skipLoop == True:
                    continue
                
                #Load second line
                LinesSimmilar = []
                LinesSimmilar.append(line1)
                for j, line2 in enumerate(lines):
                    skipLoop = False
                    line2 = line2[0]
                    if i == j:
                        continue
                    v2 = line2[2:4] - line2[0:2]
                    len2 = np.linalg.norm(v2)

                    for s in range(0, len(skipIndex)):
                        skip = skipIndex[s]
                        if skip == j:
                            skipLoop = True
                            break

                    #Skip it its too short or if it has allready been appended          
                    if len2 < minRes * minLength or skipLoop == True:
                        continue
                    
                    #Skips line if its not ~parallel
                    angleDifference = abs( calculate_line_angle(line1) - calculate_line_angle(line2) )
                    if angleDifference > maxAngleDifference:
                        continue
                        
                    #Get line points (laid it out for personaly clarity)
                    x1 = line1[0]
                    y1 = line1[1]
                    x2 = line1[2]
                    y2 = line1[3]
                    x3 = line2[0]
                    y3 = line2[1]
                    x4 = line2[2]
                    y4 = line2[3]

                    #Gives line as Ax+By=C format
                    def line(p1, p2):
                        A = (p1[1] - p2[1])
                        B = (p2[0] - p1[0])
                        C = (p1[0]*p2[1] - p2[0]*p1[1])
                        return A, B, -C

                    #Get first line as Ax+By=C format
                    L1 = line([x1, y1], [x2, y2])

                    #Gives y of an Ax+By=C line by inputting x
                    def getY(line, x):
                        y = ((line[2])-(line[0]*x))/line[1]
                        return y

                    #Input both x coords of second line. Checks if y of the extended first line meets the y of the second line
                    if abs(getY(L1, x3) - y3) > maxDistanceDifference and abs(getY(L1, x4) - y4) > maxDistanceDifference:
                        continue
                    
                    #If on same line, group together
                    skipIndex.append(j)
                    LinesSimmilar.append(line2)

                #Append each group in an array of groups
                LinesSorted.append(LinesSimmilar)



        #Go through all line groups, get min and max point (Line that spans all lines) and append
        LineFiltered = []
        for i in range(0, len(LinesSorted)):
            sorted = LinesSorted[i]

            #Initialise variables
            lineOut = sorted[0]
            minX1 = sorted[0][0] 
            minY1 = sorted[0][1] 
            maxX2 = sorted[0][2] 
            maxY2 = sorted[0][3] 

            #Gather min and max for the line group
            for j in range(0, len(sorted)):
                line = sorted[j]

                #Keep track of the mimima and maxima
                if calculate_line_angle(line) >= 0:
                    if min(line[0], line[2]) < minX1:
                        minX1 = min(line[0], line[2])
                    if max(line[0], line[2]) > maxX2:
                        maxX2 = max(line[0], line[2])
                    if min(line[1], line[3]) < minY1:
                        minY1 = min(line[1], line[3])
                    if max(line[1], line[3]) > maxY2:
                        maxY2 = max(line[1], line[3])
                if calculate_line_angle(line) < 0:
                    if min(line[0], line[2]) < minX1:
                        minX1 = min(line[0], line[2])
                    if max(line[0], line[2]) > maxX2:
                        maxX2 = max(line[0], line[2])
                    if min(line[1], line[3]) < maxY2:
                        maxY2 = min(line[1], line[3])
                    if max(line[1], line[3]) > minY1:
                        minY1 = max(line[1], line[3])

            #Fill values and append
            lineOut[0] = minX1 
            lineOut[1] = minY1
            lineOut[2] = maxX2
            lineOut[3] = maxY2
            LineFiltered.append(lineOut)

        return LineFiltered

#Get intersection points. Extend lines by 75px and check if they intersect within bounds
def get_intersection_points(LineFiltered):
    lineExtension = 75
    CornerPoints = []
    if LineFiltered is not None:

        #Get first line and extend
        for i in range(0, len(LineFiltered)):
            line1 = LineFiltered[i]
            line1 = line1
            ab = line1[2:4] - line1[0:2]
            v = (ab / np.linalg.norm(ab)) * lineExtension
            line1Ext = [line1[0:2] - v, line1[2:4] + v]

            #Get second line and extend
            for j in range(0, len(LineFiltered)):
                line2 = LineFiltered[j]
                ab = line2[2:4] - line2[0:2]
                v = (ab / np.linalg.norm(ab)) * lineExtension
                line2Ext = [line2[0:2] - v, line2[2:4] + v]
                
                #Get line as Ax+By=C format
                def line(p1, p2):
                    A = (p1[1] - p2[1])
                    B = (p2[0] - p1[0])
                    C = (p1[0]*p2[1] - p2[0]*p1[1])
                    return A, B, -C

                #Get intersection coords using determinant (if no intersection, gives [-1, -1])
                def intersection(L1, L2):
                    D  = L1[0] * L2[1] - L1[1] * L2[0]
                    Dx = L1[2] * L2[1] - L1[1] * L2[2]
                    Dy = L1[0] * L2[2] - L1[2] * L2[0]
                    if D != 0:
                        x = Dx / D
                        y = Dy / D
                        return round(x),round(y)
                    else:
                        return -1, -1

                #Get intersection point from the two lines
                L1 = line(line1Ext[0], line1Ext[1])
                L2 = line(line2Ext[0], line2Ext[1])
                R = intersection(L1, L2)

                #Calculate all the bounds and anges which are used to filter
                bound1 = ((line1Ext[0][0] <= R[0] <= line1Ext[1][0]) or (line1Ext[0][0] >= R[0] >= line1Ext[1][0]))
                bound2 = ((line2Ext[0][0] <= R[0] <= line2Ext[1][0]) or (line2Ext[0][0] >= R[0] >= line2Ext[1][0]))
                bound3 = ((line1Ext[0][1] <= R[1] <= line1Ext[1][1]) or (line1Ext[0][1] >= R[1] >= line1Ext[1][1]))
                bound4 = ((line2Ext[0][1] <= R[1] <= line2Ext[1][1]) or (line2Ext[0][1] >= R[1] >= line2Ext[1][1]))
                angle1 = calculate_line_angle(line1)
                angle2 = calculate_line_angle(line2)
                angleDifference = abs( max(angle1, angle2) - min(angle1, angle2) )
                
                height, width = OrigImage.shape[:2]

                #If exists, within bounds, at expected angle within lines and within frame, appends found corner to list
                if R and bound1 and bound2 and bound3 and bound4 and angleDifference > 12 and 0 <= R[0] <= width and 0 <= R[1] <= height:
                    append = True 
                    #Make sure not to append corners allready included (can be optimised but im too tired atm lol)
                    for k in range(0, len(CornerPoints)):
                        if CornerPoints[k] == R:
                            append = False
                    if append == True:
                        CornerPoints.append(R)

    return CornerPoints

#Compare the points to the previous frame, use only the ones that were there before (within 50px)
def average_strong_corners(CornerPoints):
    differenceXYmax = 50
    AveragePoints = []

    #Make sure to use global variables
    global PreviousPoints
    global PreviousPoints2
    global PreviousPoints3
    global PreviousPoints4

    if CornerPoints is not None:
        #Loop through all current points
        for i in range(0, len(CornerPoints)):
            point1 = CornerPoints[i]

            #Loop through frame-1 points
            for j in range(0, len(PreviousPoints)):
                point2 = PreviousPoints[j]
                xdiff = abs(point1[0] - point2[0])
                ydiff = abs(point1[1] - point2[1])

                #If current frame and frame-1 have the same relative points
                if xdiff < differenceXYmax and ydiff < differenceXYmax:
                    #Loop through frame-2 points
                    for k in range(0, len(PreviousPoints2)):
                        point3 = PreviousPoints2[k]
                        xdiff = abs(point2[0] - point3[0])
                        ydiff = abs(point2[1] - point3[1])

                        #If frame-1 and frame-2 have the same relative points
                        if xdiff < differenceXYmax and ydiff < differenceXYmax:
                            #Loop through frame-3 points
                            for l in range(0, len(PreviousPoints3)):
                                point4 = PreviousPoints3[l]
                                xdiff = abs(point3[0] - point4[0])
                                ydiff = abs(point3[1] - point4[1])

                                #If frame-2 and frame-3 have the same relative points
                                if xdiff < differenceXYmax and ydiff < differenceXYmax:
                                    #Loop through frame-4 points
                                    for m in range(0, len(PreviousPoints4)):
                                        point5 = PreviousPoints4[m]
                                        xdiff = abs(point4[0] - point5[0])
                                        ydiff = abs(point4[1] - point5[1])
                                        
                                        #If frame-3 and frame-4 have the same relative points
                                        if xdiff < differenceXYmax and ydiff < differenceXYmax:
                                            #Strong point sequence found, apply weighted average over all op the points in sequence
                                            xavg = round((point1[0]*4 + point2[0]*4 + point3[0]*3 + point4[0]*3 + point5[0]*2)/16)
                                            yavg = round((point1[1]*4 + point2[1]*4 + point3[1]*3 + point4[1]*3 + point5[1]*2)/16)

                                            #Add point to list
                                            AveragePoints.append([xavg, yavg])

    #Update all previous points                                        
    PreviousPoints4 = PreviousPoints3
    PreviousPoints3 = PreviousPoints2
    PreviousPoints2 = PreviousPoints
    PreviousPoints = CornerPoints

    return AveragePoints

# Optional function to mark points on pitch and show relative distance and octant
def mark_pts(im, pts, marker_color = (255, 50, 0), text_color = (0, 50, 255), marker_type = cv.MARKER_CROSS, text_sz = 2, marker_sz = 30):
    for i, p in enumerate(pts):
        cv.drawMarker(im, (int(p[0]), int(p[1])), marker_color, marker_type, marker_sz, 3)
        cv.putText(im, (str(p[2]) + ", " + str(int(p[3]))), (int(p[0]), int(p[1] - 10)), cv.FONT_HERSHEY_COMPLEX, text_sz, text_color, 2)
    return im

def rescale_frame(frame_input, percent=75):    
    width = int(frame_input.shape[1] * percent / 100)    
    height = int(frame_input.shape[0] * percent / 100)    
    dim = (width, height)    
    return cv.resize(frame_input, dim, interpolation=cv.INTER_AREA)

# Get point at centre of crossbar and its length
def get_goal_reference(LineGoal):
        if LineGoal:
            # sort based on y coord to pick correct marker point
            LineGoal.sort(key=itemgetter(1))

            markerPoint = 0

            # update lists of previous points/lengths for use in averaging
            setPoints.append(LineGoal[markerPoint])
            goalLengths.append(math.dist((LineGoal[markerPoint][0], LineGoal[markerPoint][1]), (LineGoal[markerPoint][2], LineGoal[markerPoint][3])))

            # only keep track of last num of points/lengths
            if len(setPoints) > 10:
                setPoints.pop(0)
                goalLengths.pop(0)

            avgX_1 = 0
            avgY_1 = 0
            avgX_2 = 0
            avgY_2 = 0
            avgLen = 0
            for i in range(0, len(setPoints)):
                avgX_1 += setPoints[i][0]
                avgY_1 += setPoints[i][1]
                avgX_2 += setPoints[i][2]
                avgY_2 += setPoints[i][3]
                avgLen += goalLengths[i]

            # get average of end points of crossbar to get centre
            bottomGoalPoint = (int(avgX_1 / len(setPoints)), int(avgY_1 / len(setPoints)))
            topGoalPoint = (int(avgX_2 / len(setPoints)), int(avgY_2 / len(setPoints)))

            centreX = (topGoalPoint[0] + bottomGoalPoint[0]) // 2
            centreY = (topGoalPoint[1] + bottomGoalPoint[1]) // 2

            setPoint = (centreX, centreY)
            goalPostLength = avgLen / len(goalLengths)
            cv.drawMarker(OrigImage, setPoint, (0, 50, 255), cv.MARKER_SQUARE, 30, 3)
            return setPoint, goalPostLength

# Takes in array of points, dest 4D array, setPoint and goalPostLength
# calcs both distance from each point to setPoint and which 'octant'
# the point is taking the setPoint as the origin and adds to new array
def calc_dist_orient(AveragePoints, averagePointsDistance, setPoint, goalPostLength):
        # Iterate over each point in AveragePoints and calculate the distance
        for l in AveragePoints:
            # Calculate the distance from setPoint
            dist = math.dist(setPoint, l)
            relDist = dist // (goalPostLength / 2)
            
            # calculate angle between two points taking setpoint as origin
            # equalise points compared to set point
            tempY = l[1] - setPoint[1]
            tempX = l[0] - setPoint[0]
            
            rads = math.atan2(tempY, tempX) / math.pi
            degrees = math.degrees(math.atan2(tempY, tempX))
            if degrees < 0:
                degrees += 360

            octant = 8 - ( degrees // 45 )

            # Concatenate the original point with the distance to form a new array with three elements
            new_point = np.concatenate((l, [relDist, octant]))
            
            # Append the new point to the list
            averagePointsDistance.append(new_point)
        return averagePointsDistance
#MAIN
while(videoCapture.isOpened()):
    ret, frame = videoCapture.read()

    if ret == True:
        TestImage = frame.copy()
        TestImageGoal = np.copy(TestImage)
        OrigImage = np.copy(TestImage)

    else: 
        break

    lower_bound = np.array([200, 200, 200], dtype=np.uint8)
    upper_bound = np.array([255, 255, 255], dtype=np.uint8)

    # Create a mask where pixels within the range are white, others are black
    TestImageGoal = cv.inRange(TestImageGoal, lower_bound, upper_bound)
    TestImageGoal = cv.cvtColor(TestImageGoal, cv.COLOR_GRAY2BGR)

    
    
    #Step 1: Apply green color segmentation (JANNAT)
    segmented_image = color_segmentation_green(TestImage)

    #TEST masking out the outside area
    filtered_image = delete_outside_area(segmented_image)

    # Step 2: Detect edges using Canny edge detection (JANNAT)
    # goal post/set point stuff
    edges, gray_image = detect_edges(TestImageGoal)
    LineGoal = getMaxLines(edges)

    edges, gray_image = detect_edges(filtered_image)
    LineFiltered = getMaxLines(edges)


    #Get intersection points. Extend lines by 75px and check if they intersect within bounds
    CornerPoints = get_intersection_points(LineFiltered)
    
    #Compare the points to the previous frame, use only the ones that were there before
    AveragePoints = average_strong_corners(CornerPoints)
        

    #Print lines
    if LineFiltered:
        #print(LineFiltered)
        for i in range(0, len(LineFiltered)):
            l = LineFiltered[i]
            cv.line(OrigImage, (l[0], l[1]), (l[2], l[3]), color[i].tolist(), 3, cv.LINE_AA)
    
    # get stable point from pitch, center of crossbar, and length of crossbar
    setPoint, goalPostLength = get_goal_reference(LineGoal)

    # create an empty list to hold the 4D points (original 2D point + distance + octant)
    averagePointsDistance = []
    calc_dist_orient(AveragePoints, averagePointsDistance, setPoint, goalPostLength)

    # sort based on distance from set point
    averagePointsDistance.sort(key=itemgetter(3))
    print(averagePointsDistance)

    mark_pts(OrigImage, averagePointsDistance)

    #Print points
    # if AveragePoints is not None:
    #     #print(AveragePoints)
    #     for i in range(0, len(AveragePoints)):
    #         p = AveragePoints[i]
    #         OrigImage = cv.circle(OrigImage, (p[0], p[1]), 20, (255, 255, 255), -1)

    #cv.imshow("Source1", segmented_image)
    #cv.imshow("Source2", eroded_edges)
    #cv.imshow("Source3", hough_image)
    cv.imshow("Source4", rescale_frame(OrigImage))
    #cv.imshow("Source5", TestMask)
    #print("Frame")

    #output_frame = OrigImage
    #output.write(output_frame) 

    if cv.waitKey(16) & 0xFF == ord('q'):
        break


#Release and destroy
videoCapture.release()
#output.release() 
cv.destroyAllWindows()