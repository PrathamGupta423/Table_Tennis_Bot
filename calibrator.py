import cv2
import numpy as np
from ball_detection import BallDetector

def detect_balls(frame, lower_hsv, upper_hsv):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    detected_balls = []


    for contour in contours:
        
        area = cv2.contourArea(contour)
        if area < 100:  # Minimum area threshold
            continue
        
        # Fit a circle to the contour
        (x,y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        
        # Filter based on circularity
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        if circularity > 0.7 and 10 < radius < 100:
            # Draw circle on the frame
            cv2.circle(frame, center, radius, (0, 255, 0), 2)
            cv2.circle(frame, center, 2, (0, 0, 255), 3)
            
            # Store ball information
            detected_balls.append((center, radius))
    
    return frame, detected_balls, mask

def nothing(x):
    pass

def main():

    frame = cv2.imread('images/camera2/image6.png')
    
    # Create a window for trackbars
    cv2.namedWindow('Color Calibration')
    
    # Create trackbars for color range adjustment
    # Hue (0-179 in OpenCV)
    cv2.createTrackbar('H Low', 'Color Calibration', 70, 179, nothing)
    cv2.createTrackbar('H High', 'Color Calibration', 130, 179, nothing)
    
    # Saturation (0-255)
    cv2.createTrackbar('S Low', 'Color Calibration', 50, 255, nothing)
    cv2.createTrackbar('S High', 'Color Calibration', 255, 255, nothing)
    
    # Value (0-255)
    cv2.createTrackbar('V Low', 'Color Calibration', 50, 255, nothing)
    cv2.createTrackbar('V High', 'Color Calibration', 255, 255, nothing)

    
    while True:
        # Read frame from video
        # ret, frame = cap.read()
        # if not ret:
        #     break
        
        # Get current trackbar positions
        h_low = cv2.getTrackbarPos('H Low', 'Color Calibration')
        h_high = cv2.getTrackbarPos('H High', 'Color Calibration')
        s_low = cv2.getTrackbarPos('S Low', 'Color Calibration')
        s_high = cv2.getTrackbarPos('S High', 'Color Calibration')
        v_low = cv2.getTrackbarPos('V Low', 'Color Calibration')
        v_high = cv2.getTrackbarPos('V High', 'Color Calibration')
        
        # Create HSV range arrays
        lower_hsv = np.array([h_low, s_low, v_low])
        upper_hsv = np.array([h_high, s_high, v_high])
        
        # Detect balls
        processed_frame, balls, mask = detect_balls(frame.copy(), lower_hsv, upper_hsv)
        
        # Display number of balls detected
        cv2.putText(processed_frame, 
                    f"Balls Detected: {len(balls)}", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 255, 0), 
                    2)
        
        # Show the processed frame and mask
        cv2.imshow('Ball Detection', processed_frame)
        cv2.imshow('Mask', mask)
        
        # Exit on 'q' key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()