import pybullet as p
import pybullet_data
import numpy as np
import time

class Simulation:

    plane = None

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
            time.sleep(1./240.)

    def disconnect():
        p.disconnect()


if __name__ == "__main__":

    Simulation.start()
    ball = Simulation.add_ball(radius=0.1 , position=[0, 1, 1])
    Simulation.simulate(1000)
    Simulation.disconnect()



    
    
