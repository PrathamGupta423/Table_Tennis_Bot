import cv2
import numpy as np

class BallDetector:

    lower_bound = np.array([0, 0, 0])
    upper_bound = np.array([255, 255, 255])


    def __init__(self, lower, upper):
        self.lower_bound = np.array(lower)
        self.upper_bound = np.array(upper)

    def detect_ball(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_bound, self.upper_bound)
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

        centers = [ball[0] for ball in detected_balls]

        return frame, detected_balls, centers
    

if __name__ == "__main__":

    frame = cv2.imread("images/camera1/image50.png")
    lower = [70,50,50]
    upper = [130,255,255]
    detector = BallDetector(lower, upper)
    frame, detected_balls, centers = detector.detect_ball(frame)
    cv2.imshow("Frame", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    

    

    