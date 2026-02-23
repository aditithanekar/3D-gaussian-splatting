import numpy as np
class Camera: 
    def __init__(self, R, T, FoVx, FoVy, image_width, image_height):
        self.R = R #rotation matrix (3d)
        self.T = T #translation matrix         #x,y,z of camera
        self.FoVx = FoVx # field of view x in radians
        self.FoVy = FoVy # field of view y in rad
        self.image_width = image_width  
        self.image_height=image_height
        self.zfar = 100.0
        self.znear = 0.01
    def world_to_camera(self, world_point):
        # we get the displacement between the world_point and the camera's location (x,y,z)
        # and then account for the rotation of the camera 
        camera_point = self.R @ (world_point - self.T) 
        return camera_point # gives us 3d position of world_point with relation to camera

    def project(self, world_point3d):
        # Projects a 3D world point to 2D pixel coordinates
        
        # get 3d world point relative to camera
        x,y,z = self.world_to_camera(world_point3d)  #split up the coordinates of the camera 3d point

        # behind the camera
        if z <= 0.0:   # ensure no div by 0 errors
            return None

        x_normalized = x / z
        y_normalized = y / z 
        
        # reference on Field of View formula: https://www.youtube.com/watch?v=pUuAx_zFnEk
        fx = self.image_width / (2 * np.tan(self.FoVx / 2))
        fy = self.image_height / (2 * np.tan(self.FoVy / 2))
        
        image_center_x = self.image_width / 2
        image_center_y = self.image_height / 2
        
        #scale the normalized x and y to 2d, and make it so center is in top left corner to avoid negatives
        pixel_x = x_normalized * fx + image_center_x
        pixel_y = y_normalized * fy  + image_center_y
        
        return pixel_x, pixel_y