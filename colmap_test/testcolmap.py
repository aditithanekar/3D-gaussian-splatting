# test_colmap_init.py
import pycolmap as colmap
from gaussian3d import Gaussian3D
import numpy as np
from camera import Camera
from PIL import Image
import matplotlib.pyplot as plt
from main import render_multiple_gaussians  # render function

def test_initialization():
    # Load COLMAP output created
    reconstruction = colmap.Reconstruction("sparse/0")
        
    # Initialize Gaussians from the COLMAP output point cloud
    gaussians = []
    for point_id, point in reconstruction.points3D.items():
        g = Gaussian3D(
            center=point.xyz,
            scale=np.array([0.1, 0.1, 0.1]),
            rotation=np.array([1.0, 0.0, 0.0, 0.0]),
            color=point.color / 255.0,
            opacity=0.9
        )
        gaussians.append(g)

    print(f"Initialized {len(gaussians)} Gaussians from COLMAP")

    return gaussians, reconstruction

def load_cameras_from_colmap(reconstruction, images_path="images"):
    # Extract camera info from COLMAP
    
    cameras_data = []
    
    for img_id, colmap_image in reconstruction.images.items():
        # Get camera intrinsics
        colmap_cam = reconstruction.cameras[colmap_image.camera_id]
        
        pose = colmap_image.cam_from_world() 
    
        if pose is None:
            print(f"Skipping unregistered image: {colmap_image.name}")
            continue  # or handle as needed
        
        R = pose.rotation.matrix()   # 3x3 world-to-camera rotation
        t = pose.translation         # (3,) translation
        
        # Compute field of view from focal length
        fx = colmap_cam.params[0]  # focal length (depends on camera model)
        # Inside the loop, after getting colmap_cam
        fx_raw = colmap_cam.params[0]
        fy_raw = colmap_cam.params[1] if len(colmap_cam.params) > 1 else fx_raw
        
        width = colmap_cam.width
        height = colmap_cam.height
        cx_raw = colmap_cam.params[2] if len(colmap_cam.params) > 2 else width / 2
        cy_raw = height / 2
        
        FoVx = 2 * np.arctan(width / (2 * fx))
        FoVy = 2 * np.arctan(height / (2 * fx))  # Assuming square pixels
        
        # Load target image
        img_path = f"{images_path}/{colmap_image.name}"
        target_img = np.array(Image.open(img_path)).astype(np.float32) / 255.0
        
        camera_center = -R.T @ t  # actual camera position in world space
        # Create Camera object
        cam = Camera(
            R=R,
            T=camera_center, #changed this
            FoVx=FoVx,
            FoVy=FoVy,
            image_width=width,
            image_height=height, 
            fx_raw=fx_raw,
            fy_raw=fy_raw,
            cx_raw=cx_raw,
            cy_raw=cy_raw
        )
        
        cameras_data.append({
            'camera': cam,
            'target_image': target_img,
            'name': colmap_image.name
        })
    print(f"Image: {colmap_image.name} | Model: {colmap_cam.model} | Params: {colmap_cam.params}")
    print(f"Loaded {len(cameras_data)} cameras with target images")
    for i, cam_data in enumerate(cameras_data):
        cam = cam_data['camera']
        R = np.array(cam.R)
        T = np.array(cam.T)  # now these are actual world positions
        forward = R[2, :]
    print(f"Cam {i}: center={np.round(T, 2)}, forward={np.round(forward, 2)}")
    return cameras_data



def test_initial_render(gaussians, cameras_data):
    # Render initial Gaussians and compare to target
    
    # Pick first camera
    cam_data = cameras_data[1] # picking second camera now
    camera = cam_data['camera']
    target = cam_data['target_image']
    
    print(f"\nRendering from: {cam_data['name']}")
    print(f"Image size: {camera.image_width}x{camera.image_height}")
    print(f"Number of Gaussians: {len(gaussians)}")
    
    # Render initial Gaussians
    print("Rendering...")
    rendered = render_multiple_gaussians(
        gaussians,
        camera,
        (camera.image_width), 
        (camera.image_height)
    )
    
    # Compute initial loss
    loss = np.mean(np.abs(target - rendered))
    print(f"Initial L1 loss: {loss:.4f}")
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(target)
    axes[0].set_title("Target Image")
    axes[0].axis('off')
    
    axes[1].imshow(np.clip(rendered, 0, 1))
    axes[1].set_title(f"Rendered (Initial)\nLoss: {loss:.4f}")
    axes[1].axis('off')
    
    axes[2].imshow(np.abs(target - rendered))
    axes[2].set_title("Absolute Difference")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig("initial_render.png")
    plt.show()
    
    print("Visualization saved as 'initial_render.png'")

# # Test it:
# gaussians, recon = test_initialization()
# cameras_data = load_cameras_from_colmap(recon)

# # # Run the test
# test_initial_render(gaussians, cameras_data)
