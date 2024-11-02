# -*- coding: utf-8 -*-
"""Project_vedio.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1YORe7TbQt-wA-rjsZEw90qZMluJhpWr7
"""

# Import libraries here
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from google.colab.patches import cv2_imshow

"""# Add all the functions here"""

# Function for color segmentation (only green color)
def color_segmentation_green(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([90, 255, 255])
    mask_green = cv2.inRange(hsv_image, lower_green, upper_green)
    segmented_image = cv2.bitwise_and(image, image, mask=mask_green)
    return segmented_image

# Function to detect edges using Canny edge detection
def detect_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)     # Apply Gaussian blur to reduce noise
    edges = cv2.Canny(blurred, 30, 90)             # Canny edge detection
    return edges, gray

# Function to detect lines using Hough Line Transform
def detect_hough_lines(edges):
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=150, maxLineGap=10)
    return lines

# Function to calculate the angle between two vectors
def calculate_angle_between_vectors(vector1, vector2):
    # Normalize the vectors
    norm_vector1 = vector1 / np.linalg.norm(vector1)
    norm_vector2 = vector2 / np.linalg.norm(vector2)

    # Calculate the dot product
    dot_product = np.clip(np.dot(norm_vector1, norm_vector2), -1.0, 1.0)
    angle_rad = np.arccos(dot_product)  # Angle in radians
    angle_deg = np.degrees(angle_rad)   # Convert to degrees
    return angle_deg

# Function to find the longest line
def find_longest_line(lines):
    longest_line = None
    max_length = 0
    if lines is not None:
        for line in lines:
            length = np.sqrt((line[0][2] - line[0][0])**2 + (line[0][3] - line[0][1])**2)
            if length > max_length:
                max_length = length
                longest_line = line
    return longest_line

# Function to find and draw parallel and perpendicular lines
def find_and_draw_parallel_perpendicular(image, lines, longest_line):
    combined_image = image.copy()
    parallel_threshold = 10  # Threshold for parallelism
    perpendicular_threshold_low = 0 # Threshold for perpendicularity
    perpendicular_threshold_high = 80 # Threshold for perpendicularity


    if longest_line is not None:
        vector_longest = np.array([longest_line[0][2] - longest_line[0][0], longest_line[0][3] - longest_line[0][1]])

        for line in lines:
            vector_current = np.array([line[0][2] - line[0][0], line[0][3] - line[0][1]])
            angle = calculate_angle_between_vectors(vector_longest, vector_current)

            # Optional
            # Put the angle text near the midpoint of the line
            #mid_x = (line[0][0] + line[0][2]) // 2
            #mid_y = (line[0][1] + line[0][3]) // 2

           # Overlay angle text
            #angle_text = f"{int(angle)}"  # Remove the degree symbol
            #cv2.putText(combined_image, angle_text, (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

            if angle <= parallel_threshold:
              # Draw parallel lines in green
               cv2.line(combined_image, tuple(line[0][:2]), tuple(line[0][2:]), (255, 0, 0), 2)


            if perpendicular_threshold_low <=(90-angle) <= perpendicular_threshold_high:
                # Draw perpendicular lines in red
                cv2.line(combined_image, tuple(line[0][:2]), tuple(line[0][2:]), (255, 0, 0), 2)

    return combined_image

# Kim's code: please check if this function work as before
def corner_detection(lines):
    # Values for filtering and grouping
    skipIndex = []
    LinesSorted = []
    line_extension=75
    minLength=10
    minRes=1
    maxAngleDifference=5,
    maxDistanceDifference=15
    # Filter and group lines
    if lines is not None:
        for i, line1 in enumerate(lines):
            line1 = line1[0]
            v1 = line1[2:4] - line1[0:2]
            len1 = np.linalg.norm(v1)

            if any(skip == i for skip in skipIndex) or len1 < minRes * minLength:
                continue

            LinesSimmilar = [line1]
            for j, line2 in enumerate(lines):
                if i == j:
                    continue

                line2 = line2[0]
                v2 = line2[2:4] - line2[0:2]
                len2 = np.linalg.norm(v2)

                if any(skip == j for skip in skipIndex) or len2 < minRes * minLength:
                    continue

                angleDifference = abs(calculate_line_angle(line1) - calculate_line_angle(line2))
                if angleDifference > maxAngleDifference:
                    continue

                x1, y1, x2, y2 = line1
                x3, y3, x4, y4 = line2

                def calculate_line(p1, p2):
                    A = (p1[1] - p2[1])
                    B = (p2[0] - p1[0])
                    C = (p1[0] * p2[1] - p2[0] * p1[1])
                    return A, B, -C

                L1 = calculate_line([x1, y1], [x2, y2])

                def getY(line, x):
                    return (line[2] - line[0] * x) / line[1]

                if abs(getY(L1, x3) - y3) > maxDistanceDifference and abs(getY(L1, x4) - y4) > maxDistanceDifference:
                    continue

                skipIndex.append(j)
                LinesSimmilar.append(line2)

            LinesSorted.append(LinesSimmilar)

    # Filtered lines
    LineFiltered = []
    for sorted_lines in LinesSorted:
        lineOut = sorted_lines[0]
        minX1, minY1 = sorted_lines[0][:2]
        maxX2, maxY2 = sorted_lines[0][2:]

        for line in sorted_lines:
            if calculate_line_angle(line) >= 0:
                minX1 = min(minX1, min(line[0], line[2]))
                maxX2 = max(maxX2, max(line[0], line[2]))
                minY1 = min(minY1, min(line[1], line[3]))
                maxY2 = max(maxY2, max(line[1], line[3]))
            else:
                minX1 = min(minX1, min(line[0], line[2]))
                maxX2 = max(maxX2, max(line[0], line[2]))
                maxY2 = min(maxY2, min(line[1], line[3]))
                minY1 = max(minY1, max(line[1], line[3]))

        lineOut[:2] = minX1, minY1
        lineOut[2:] = maxX2, maxY2
        LineFiltered.append(lineOut)

    CornerPoints = []
    if LineFiltered:
        for i, line1 in enumerate(LineFiltered):
            ab = line1[2:4] - line1[0:2]
            v = (ab / np.linalg.norm(ab)) * line_extension
            line1Ext = [line1[0:2] - v, line1[2:4] + v]

            for j, line2 in enumerate(LineFiltered):
                ab = line2[2:4] - line2[0:2]
                v = (ab / np.linalg.norm(ab)) * line_extension
                line2Ext = [line2[0:2] - v, line2[2:4] + v]
                def calculate_line(p1, p2):
                    A = (p1[1] - p2[1])
                    B = (p2[0] - p1[0])
                    C = (p1[0] * p2[1] - p2[0] * p1[1])
                    return A, B, -C
                L1 = calculate_line(line1Ext[0], line1Ext[1])
                L2 = calculate_line(line2Ext[0], line2Ext[1])

                def intersection(L1, L2):
                    D = L1[0] * L2[1] - L1[1] * L2[0]
                    Dx = L1[2] * L2[1] - L1[1] * L2[2]
                    Dy = L1[0] * L2[2] - L1[2] * L2[0]
                    if D != 0:
                        return round(Dx / D), round(Dy / D)
                    else:
                        return -1, -1

                R = intersection(L1, L2)

                bound1 = ((line1Ext[0][0] <= R[0] <= line1Ext[1][0]) or (line1Ext[0][0] >= R[0] >= line1Ext[1][0]))
                bound2 = ((line2Ext[0][0] <= R[0] <= line2Ext[1][0]) or (line2Ext[0][0] >= R[0] >= line2Ext[1][0]))
                bound3 = ((line1Ext[0][1] <= R[1] <= line1Ext[1][1]) or (line1Ext[0][1] >= R[1] >= line1Ext[1][1]))
                bound4 = ((line2Ext[0][1] <= R[1] <= line2Ext[1][1]) or (line2Ext[0][1] >= R[1] >= line2Ext[1][1]))

                if R != (-1, -1) and bound1 and bound2 and bound3 and bound4:
                    CornerPoints.append(R)

    return CornerPoints

# Function to calculate the angle of a line
def calculate_line_angle(line):

    x1, y1, x2, y2 = line #was [0]

    angle_radians = np.arctan2(y2 - y1, x2 - x1)

    angle_degrees = np.degrees(angle_radians)

    return angle_degrees

"""# Processsing vedio frames"""

# video process function
def main(input_video_file: str, output_video_file: str) -> None:
    # OpenCV video objects to work with
    cap = cv2.VideoCapture(input_video_file)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    # Initialize a list to store corner positions
    all_corners = []  # This will store the corners from all frames


    # Get video properties
    fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Saving output video as .mp4
    out = cv2.VideoWriter(output_video_file, fourcc, fps, (frame_width, frame_height))

    frame_counter = 0  # Initialize frame counter

    while cap.isOpened():  # Loop until the video ends
        ret, frame = cap.read()  # Read the next frame
        if not ret:
            break  # Exit the loop if there are no frames to read

        # Step 1: Apply green color segmentation
        segmented_image = color_segmentation_green(frame)

        # Step 2: Detect edges using Canny edge detection
        edges, gray_image = detect_edges(segmented_image)

        # Step 3: Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        eroded_edges = cv2.erode(dilated_edges, kernel, iterations=1)

        # Step 4: Detect lines using Hough Line Transform
        lines = detect_hough_lines(eroded_edges)
        hough_image = frame.copy()

        # Step 5: Find the longest line
        longest_line = find_longest_line(lines)

        # Step 6: Draw combined image with both parallel and perpendicular lines
        combined_image = find_and_draw_parallel_perpendicular(hough_image, lines, longest_line)

        # Step 7: Corner detection
        CornerPoints = corner_detection(lines)  # Call your corner detection function
        all_corners.extend(CornerPoints)  # Save corner points for this frame

        # Draw corners on the combined image with thick dots
        for corner in CornerPoints:
          cv2.circle(combined_image, (corner[0], corner[1]), 10, (255, 255, 255), -1)  # Draw corners in white

        # optional: print all corners
        #print("Detected Corner Points:")
        #for corner in all_corners:
          #print(corner)  # Ensure this is properly indented

        # Optional: Display image with all parallel and perpendicular lines and angles
        plt.figure(figsize=(10, 10))
        plt.title("All  Lines with corners")
        plt.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

        # Step 8: Call Homography function here (not implemented yet)





        # Write the processed frame to the output video
        out.write(combined_image)  # chage variable here
        frame_counter += 1

        # Optional: Display the processed image every 40 frames
        if frame_counter % 50== 0:
            cv2_imshow(combined_image)  # change variable here
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break  # Exit if 'q' is pressed

    cap.release()
    out.release()
    cv2.destroyAllWindows()

"""
# Input and output video

"""

# Path to your video file
video_path = '/content/drive/MyDrive/Image processing assignments /project/InputVideoCOMP (online-video-cutter.com).mp4'
output_path = '/content/drive/MyDrive/Image processing assignments /project/output_video.mp4'
main(video_path, output_path)