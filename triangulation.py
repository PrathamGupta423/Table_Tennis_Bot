import cv2 
import numpy as np
from ball_detection import BallDetector

def triangulate(cam1_parameters, cam2_parameters, detector, img1, img2):

    cam1_coords = cam1_parameters["position"]
    cam2_coords = cam2_parameters["position"]

    cam1_view_matrix = cam1_parameters["view_matrix"]
    cam1_projection_matrix = cam1_parameters["projection_matrix"]

    cam2_view_matrix = cam2_parameters["view_matrix"]
    cam2_projection_matrix = cam2_parameters["projection_matrix"]

    cam1_img = img1
    cam2_img = img2

    cam1_img, detected_balls1, centers1 = detector.detect_ball(cam1_img)
    cam2_img, detected_balls2, centers2 = detector.detect_ball(cam2_img)

    if len(detected_balls1) == 0 or len(detected_balls2) == 0:
        return None
    
    ball1 = detected_balls1[0]
    ball2 = detected_balls2[0]
    
    ball1_center = ball1[0]
    ball2_center = ball2[0]

    ball1_center = np.array([ball1_center[0], ball1_center[1]])
    ball2_center = np.array([ball2_center[0], ball2_center[1]])

    # Modify projection matrices to 3x4 by taking the first 3 rows
    cam1_projection_matrix = cam1_projection_matrix[:3, :]
    cam2_projection_matrix = cam2_projection_matrix[:3, :]

    # Triangulate the 3D position of the ball
    ball1_3d = cv2.triangulatePoints(
        cam1_projection_matrix, 
        cam2_projection_matrix, 
        ball1_center, 
        ball2_center
    )

    # Convert to float for division
    ball1_3d = ball1_3d.astype(np.float64)

    # Normalize the homogeneous coordinates (divide by the w component)
    ball1_3d /= ball1_3d[3]

    # Reshape cam1_view_matrix to a 4x4 matrix if it's 1D
    cam1_view_matrix = cam1_view_matrix.reshape(4, 4)

    # Convert the 3D position to world coordinates (in homogeneous coordinates)
    ball1_3d_world = np.dot(cam1_view_matrix, ball1_3d[:4])  # Use all 4 components of ball1_3d

    # Normalize the result (if necessary)
    ball1_3d_world /= ball1_3d_world[3]

    return ball1_3d_world[:3]  # Return the 3D coordinates (ignoring the homogeneous coordinate)




if __name__ == "__main__":

    cam1 = {
        "view_matrix": np.array([1.0, 0.0, -0.0, 0.0, -0.0, 0.7071068286895752, -0.7071067690849304, 0.0, 0.0, 0.7071067690849304, 0.7071068286895752, 0.0, -0.0, -0.0, -2.8284270763397217, 1.0]),

        "projection_matrix": np.array([[1.299038052558899, 0.0, 0.0, 0.0], [0.0, 1.7320507764816284, 0.0, 0.0], [0.0, 0.0, -1.0020020008087158, -1.0], [0.0, 0.0, -0.20020020008087158, 0.0]]),
        "position": [0, -2, 2]
    }

    cam2 = {
        "view_matrix": np.array([-1.0, 6.181723932741079e-08, -6.181723932741079e-08, 0.0, -8.742277657347586e-08, -0.7071068286895752, 0.7071067690849304, 0.0, -3.552713678800501e-15, 0.7071067690849304, 0.7071068286895752, 0.0, 7.105427357601002e-15, -0.0, -2.8284270763397217, 1.0]),
        "projection_matrix": np.array([[1.299038052558899, 0.0, 0.0, 0.0], [0.0, 1.7320507764816284, 0.0, 0.0], [0.0, 0.0, -1.0020020008087158, -1.0], [0.0, 0.0, -0.20020020008087158, 0.0]]),
        "position": [0, 2, 2]
    }

    detector = BallDetector([70,50,50], [130,255,255])

    img1 = cv2.imread("images/camera1/image50.png")
    img2 = cv2.imread("images/camera2/image50.png")

    ball_3d = triangulate(cam1, cam2, detector, img1, img2)

    print(ball_3d)


