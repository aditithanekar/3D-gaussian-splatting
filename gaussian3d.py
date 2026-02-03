import numpy as np
import math
class Gaussian3D:
    def __init__(self, mean, center, scale, rotation, color, opacity):
        self.center = np.asarray(center) #mu is centered at point (u1, u2, u3)
        self.scale = np.asarray(scale)    #3d vector s (s1, s2, s3)
        self.rotation = np.asarray(rotation)   #4d quaternion q (q_r, q_i, q_j, q_k)
        self.color = np.asarray(color)   # (3,)
        self.alpha = opacity #a is opacity
    def get_covariance(self):
        # if R is rotation matrix
        # and S is scaling matrix
        #covariance Sigma = R * S * S^T * R^T 
        # we need to make sure that q(rotation) is normalized 
        # and the scale and rotation vectors can be converted into their matrix form !!!! TODO
        scale_matrix = np.diag(self.scale) # turning 3d vec scale into 3d matrix diagonal scale matrix
        
        #normalize the quaternion rotation 
        q0,q1,q2,q3 = self.rotation
        mag_rotation = math.sqrt(q0**2 + q1**2 + q2**2 + q3**2) # get the magnitude for normalizing
        normalized_rotation = self.rotation/mag_rotation  #each element divided by the magnitude
        q0,q1,q2,q3 = normalized_rotation
        
        # convert normalized_rotation to be a matrix instead of 4d vector: reference: https://automaticaddison.com/how-to-convert-a-quaternion-to-a-rotation-matrix/
        # First row of the rotation matrix
        r00 = 2 * (q0 * q0 + q1 * q1) - 1
        r01 = 2 * (q1 * q2 - q0 * q3)
        r02 = 2 * (q1 * q3 + q0 * q2)
        
        # Second row of the rotation matrix
        r10 = 2 * (q1 * q2 + q0 * q3)
        r11 = 2 * (q0 * q0 + q2 * q2) - 1
        r12 = 2 * (q2 * q3 - q0 * q1)
        
        # Third row of the rotation matrix
        r20 = 2 * (q1 * q3 - q0 * q2)
        r21 = 2 * (q2 * q3 + q0 * q1)
        r22 = 2 * (q0 * q0 + q3 * q3) - 1
        
        # 3x3 rotation matrix
        rot_matrix = np.array([[r00, r01, r02],
                            [r10, r11, r12],
                            [r20, r21, r22]])
            
            
        return rot_matrix @ scale_matrix @ scale_matrix.T @ rot_matrix.T # if it's a numpy array you can use .T to get transpose, @ for matmul