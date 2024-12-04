import pybullet as p
import pybullet_data
import numpy as np
import time
import os
import cv2

class Simulation:

    plane = None
    camera_list = []

    def start():
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        plane = p.loadURDF("plane.urdf")

    def add_ball(radius=0.5 , mass=1, position= [0, 0, 1], color= [1, 0, 0, 1]):
        ball_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
        ball_visual = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
        ball = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=ball_collision, baseVisualShapeIndex=ball_visual, basePosition=position)
        p.changeDynamics(ball, -1, restitution=0.9)
        return ball
    
    def simulate(iter = 100):
        for i in range(iter):
            p.stepSimulation()
            
            for j , camera in enumerate(Simulation.camera_list):
                Simulation.save_image(j, "image" + str(i))
            time.sleep(1./240.)

    def disconnect():
        p.disconnect()

    def add_camera(width, height, view_matrix, projection_matrix, image_folder):

        if not os.path.exists(image_folder):
            os.makedirs(image_folder)

        cam = {
            "view_matrix": view_matrix,
            "projection_matrix": projection_matrix,
            "image_folder": image_folder,
            "width": width,
            "height": height
        }
        Simulation.camera_list.append(cam)

    def generate_view_matrix(target_position = [0,0,0], distance=2, yaw=45, pitch=-30, roll=0 , upaxisIndex=2):
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=target_position,
            distance=distance,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            upAxisIndex=upaxisIndex
        )
        return view_matrix
    
    def generate_projection_matrix(fov = 60, aspect = 640/480, near = 0.1, far= 100):
        projection_matrix = p.computeProjectionMatrixFOV(
            fov = fov,
            aspect = aspect,
            nearVal = near,
            farVal = far
        )
        return projection_matrix
    
    def get_camera_image(camera_id = 0):
        camera = Simulation.camera_list[camera_id]
        img = p.getCameraImage(
            width=camera["width"], 
            height=camera["height"], 
            viewMatrix=camera["view_matrix"],
            projectionMatrix=camera["projection_matrix"]
        )
        width=camera["width"]
        height=camera["height"]

        img = np.reshape(img[2], (height, width, 4))[:,:,:3]
    


        return img
    
    def save_image(camera_id,name):
        camera = Simulation.camera_list[camera_id]
        img = Simulation.get_camera_image(camera_id)
        folder = camera["image_folder"]
        cv2.imwrite(folder + "/" + name + ".png", img)
    
        


if __name__ == "__main__":

    Simulation.start()
    ball = Simulation.add_ball(radius=0.1 , position=[0, 0.3, 1])
    ball_green = Simulation.add_ball(radius=0.05 , position=[0, 0, 0], color=[0, 1, 0, 1])
    ball_blue = Simulation.add_ball(radius=0.05 , position=[1, 0, 0], color=[0, 0, 1, 1])
    ball_black = Simulation.add_ball(radius=0.05 , position=[0.5, 0.5, 0], color=[0, 0, 0, 1])


    d = 2*np.sqrt(2)

    view_matrix = Simulation.generate_view_matrix(target_position=[0, 0, 0], distance=d, yaw=0, pitch=-45, roll=0 , upaxisIndex=2)
    print(view_matrix)
    projection_matrix = Simulation.generate_projection_matrix()
    image_folder = "images/camera1"
    Simulation.add_camera(640,480,view_matrix, projection_matrix, image_folder)

    view_matrix = Simulation.generate_view_matrix(target_position=[0, 0, 0], distance=d, yaw=180, pitch=-45, roll=0 , upaxisIndex=2)
    print(view_matrix)
    projection_matrix = Simulation.generate_projection_matrix()
    image_folder = "images/camera2"
    Simulation.add_camera(640,480,view_matrix, projection_matrix, image_folder)

    Simulation.simulate(100)
    Simulation.disconnect()



    
    
