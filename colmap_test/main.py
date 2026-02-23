from camera import Camera
import numpy as np 
from gaussian3d import Gaussian3D
import matplotlib.pyplot as plot
import math

def render_multiple_gaussians(gaussians, camera, image_width, image_height):
    image = np.zeros((image_height, image_width, 3), dtype=np.float32)
    transmittance = np.ones((image_height, image_width), dtype=np.float32) #initialize transmittance
    
    gaussians_sorted_by_depth = []
    #sort the gaussians by their depths to end with closer ones and start with farther
    for gaussian in gaussians: 
        #get the depth (z)
        depth = camera.world_to_camera(gaussian.center)[2] # depth is the z component of this 3d camera space coord
        gaussians_sorted_by_depth.append((depth, gaussian))
    gaussians_sorted_by_depth.sort(key=lambda x: x[0], reverse=True) # sort it with the farthest first(descending order)
    
    
    for depth, gaussian in gaussians_sorted_by_depth:
        pixel = gaussian.project_point(camera) # get the gaussian in 2d pixel coords
        if pixel is None:
            continue # just leave it out of the computation bc it could be behind the camera
        pixel_x, pixel_y = pixel
        
        print(f"Pixel is: {pixel_x:.2f}, {pixel_y:.2f}")
        covariance_camera_sigma_prime = gaussian.project_covariance_to_camera_coords(camera)
        # print("covariance is" )
        # print(covariance_camera_sigma_prime)
        print("splatting gaussian")
    
        cov_inv = np.linalg.inv(covariance_camera_sigma_prime)
        
        # axis-aligned bounding box
        sigma_x = math.sqrt(covariance_camera_sigma_prime[0,0])
        sigma_y = math.sqrt(covariance_camera_sigma_prime[1,1])
        radius = max(sigma_x, sigma_y) * 3.5   # 3.0–4.0 is typical trade-off
        
        # Compute bounding box (clipped to image)
        x_min = max(0, int(pixel_x - radius))
        x_max = min(image_width - 1, int(pixel_x + radius + 0.999))
        y_min = max(0, int(pixel_y - radius))
        y_max = min(image_height - 1, int(pixel_y + radius + 0.999))
        
        if x_min > x_max or y_min > y_max:
            continue  # completely outside screen
        
        #iterate through the 2d pixel grid (this is kinda inefficient so we should use bounding box)
        for y in range(y_min, y_max + 1):
            for x in range(x_min, x_max + 1):
                #splat
                offset = np.array([x - pixel_x, y - pixel_y]) # how much is the gaussian 2d pixel offset from each pixel
                gaussian_weight_at_point = math.exp(-0.5 * ((offset).T @ cov_inv @ offset))                
                alpha = gaussian.alpha * gaussian_weight_at_point
                
                # alpha blending:
                image[y, x] += transmittance[y, x] * alpha * gaussian.color
                transmittance[y, x] *= (1.0 - alpha)
                        
    return image
def main():
    rotation = np.eye(3) # identity matrix, don't want any rotation yet
    translation1 = [0,0,0]
    translation2 = [-1,0,0]
    translation3 = [0,0,1]
    translation4 = [0,0,-1]
    FoVx = np.deg2rad(60)
    FoVy = np.deg2rad(45)

    image_width = 800
    image_height = 600

    cam1 = Camera(
        R=rotation,
        T=translation1,
        FoVx=FoVx,
        FoVy=FoVy,
        image_width=image_width,
        image_height=image_height
    )
    
    gaussians = [ 
        Gaussian3D(
        center=np.array([0, 0, 5]),   # in front of camera
        scale=np.array([0.5, 1.0, 0.5]),
        rotation=[np.cos(np.pi/8), 0, 0, np.sin(np.pi/8)],
        color=np.array([0.8, 0.0, 0.1]), # color of gaussian rgb
        opacity=1.0), # full opacity
        Gaussian3D(
            center=np.array([0, 0, 8]),     # Far
            scale=np.array([2.0, 2.0, 2.0]),
            rotation=[1, 0, 0, 0],
            color=np.array([1.0, 0.0, 0.0]),  # Red
            opacity=0.8
        ),
        Gaussian3D(
            center=np.array([0.5, 0, 5]),   # Middle
            scale=np.array([1.5, 1.5, 1.5]),
            rotation=[1, 0, 0, 0],
            color=np.array([0.0, 1.0, 0.0]),  # Green
            opacity=0.8
        ),
        Gaussian3D(
            center=np.array([-0.5, 0, 3]),  # Near
            scale=np.array([1.0, 1.0, 1.0]),
            rotation=[1, 0, 0, 0],
            color=np.array([0.0, 0.0, 1.0]),  # Blue
            opacity=0.8
        )]
    
    image = render_multiple_gaussians(gaussians, cam1, image_width=image_width, image_height=image_height )        
    print("showing plot")     

    plot.imshow(np.clip(image, 0, 1))
    plot.axis("off")
    plot.show()
    pass


if __name__ == "__main__":
    main()
    
