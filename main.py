from camera import Camera
import numpy as np 
from gaussian3d import Gaussian3D
import matplotlib.pyplot as plot
import math


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

    image = np.zeros((image_height, image_width, 3), dtype=np.float32)
    
    
    gaussian = Gaussian3D(
        center=np.array([0, 0, 5]),   # in front of camera
        scale=np.array([0.5, 1.0, 0.5]),
        rotation=[np.cos(np.pi/8), 0, 0, np.sin(np.pi/8)],
        color=np.array([0.1, 0.5, 0.0]), # color of gaussian rgb
        opacity=1.0) # full opacity
    
    print("projecting pt")
    pixel = gaussian.project_point(cam1) # get the gaussian in 2d pixel coords
    
    if pixel is None:
        print("Error, no 2D pixel from projected gaussian point")
        return
    pixel_x, pixel_y = pixel
    print("Pixel is: ")
    print(pixel)

    covariance_camera_sigma_prime = gaussian.project_covariance_to_camera_coords(cam1)
    print("covariance is" )
    print(covariance_camera_sigma_prime)
    print("splatting gaussian")
    
    cov_inv = np.linalg.inv(covariance_camera_sigma_prime)


    #iterate through the 2d pixel grid
    for y in range(image_height):
        for x in range(image_width):
            #splat
            offset = np.array([x - pixel_x, y - pixel_y]) # how much is the gaussian 2d pixel offset from each pixel
            gaussian_weight_at_point = math.exp(-0.5 * ((offset).T @ cov_inv @ offset))
            image[y,x] += gaussian.alpha * gaussian_weight_at_point * gaussian.color
            
    print("showing plot")     

    plot.imshow(np.clip(image, 0, 1))
    plot.axis("off")
    plot.show()
    pass


if __name__ == "__main__":
    main()
    
