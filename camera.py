class Camera: 
    def __init__(self, R, T, FoVx, FoVy, image_width, image_height):
        self.R = R #rotation matrix
        self.T = T #translation matrix         #x,y,z of camera
        self.FoVx = FoVx # field of view x in radians
        self.FoVy = FoVy # field of view y in rad
        self.image_width = image_width  
        self.image_height=image_height
        self.zfar = 100.0
        self.znear = 0.01
    
