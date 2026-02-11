import numpy as np
import math
class Gaussian3D:
    def __init__(self, center, scale, rotation, color, opacity):
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
        # and the scale and rotation vectors can be converted into their matrix form !
        
        scale_matrix = np.diag(self.scale) # turning 3d vec scale into 3d matrix diagonal scale matrix
        
        #normalize the quaternion rotation 
        q0,q1,q2,q3 = self.rotation
        mag_rotation = math.sqrt(q0**2 + q1**2 + q2**2 + q3**2) # get the magnitude for normalizing
        
        # make sure that you don't get dividing by small almost zero number errors
        if mag_rotation < 1e-8:
            raise ValueError("Quaternion magnitude too small")
        
        normalized_rotation = self.rotation/mag_rotation  #each element divided by the magnitude
        q_r, q_i, q_j, q_k = normalized_rotation
        
        
        # convert normalized_rotation to be a matrix instead of 4d vector
        # can see Equation 10 on page 13 of the paper 
        # First row of the rotation matrix
        r00 = 1 - (2 * (q_j * q_j + q_k * q_k)) 
        r01 = 2 * (q_i * q_j - q_r * q_k)
        r02 = 2 * (q_i * q_k + q_r * q_j)
        
        # Second row of the rotation matrix
        r10 = 2 * (q_i * q_j + q_r * q_k)
        r11 = 1 - (2 * (q_i * q_i + q_k * q_k))
        r12 = 2 * (q_j * q_k - q_r * q_i)
        
        # Third row of the rotation matrix
        r20 = 2 * (q_i * q_k - q_r * q_j)
        r21 = 2 * (q_j * q_k + q_r * q_i)
        r22 = 1 - (2 * (q_i * q_i + q_j * q_j))

        
        # 3x3 rotation matrix
        rot_matrix = np.array([[r00, r01, r02],
                            [r10, r11, r12],
                            [r20, r21, r22]])
            
        #Sigma in the formula    
        covariance = rot_matrix @ scale_matrix @ scale_matrix.T @ rot_matrix.T # if it's a numpy array you can use .T to get transpose, @ for matmul
        return covariance
    def compute_jacobian(self, camera):
        #Compute Jacobian of perspective projection.
        #J is 2x3 matrix mapping 3D camera coords to 2D screen coords
        
        # Get point in camera coordinates
        cam_point = camera.world_to_camera(self.center)
        x, y, z = cam_point
        
        # Focal lengths from FoV
        fx = camera.image_width / (2 * np.tan(camera.FoVx / 2))
        fy = camera.image_height / (2 * np.tan(camera.FoVy / 2))
        
        # Jacobian of perspective projection
        J = np.array([
            [fx / z,      0,    -fx * x / (z**2)],
            [0,      fy / z,    -fy * y / (z**2)]
        ])
        
        return J  # 2x3 matrix
    def project_covariance_to_camera_coords(self, camera):
        # projecting 3d gaussians to 2d for rendering 
        # Sigma' = J W Sigma W^T J^T
        # J is the Jacobian of the affine approximation of the projective transformation 
        # W is a given viewing transformation
        # Sigma is covariance
        # Sigma'(prime) is covariance matrix in camera coordinates
        
        Jacobian = self.compute_jacobian(camera)
        
        W = camera.R  #rotation defines the viewing transformation
        
        Sigma = self.get_covariance() # Sigma = covariance
        
        covariance_camera = Jacobian @ W @ Sigma @ W.T @ Jacobian.T
        return covariance_camera
    
    def project_point(self, camera):
        return camera.project(self.center) # project to 2d pixel for the center of the gaussian
    #fix this later
    # def idk(self, viewing_transformation_in):
    #     # G(x) = e^(0.5f(x)^T * Sigma' * x)
    #     # x is the center point mu aka the mean 
    #     sigma_prime = self.project_covariance_to_camera_coords(self, viewing_transformation=viewing_transformation_in)
    #     gaussian_weight_at_point = math.exp(-0.5 * (self.center).T @ sigma_prime @ self.center)

    def alpha_blend(self, transmittance, alpha, color):
        # get N somehow 
        N = 10
        alpha_blended_color = 0.0 
        for i in range(N):
            alpha_blended_color += transmittance[i]*alpha[i]*color[i] 
        return alpha_blended_color
        
        # how do you get N? the amount of points that are overlapping on the pixel? 
    
    
    #next steps: 
    # use colmap --> sfm points, make a point cloud, iterate through that cloud and make each of them gaussians, then alpha blend 
    # then after that you have to do training? 
