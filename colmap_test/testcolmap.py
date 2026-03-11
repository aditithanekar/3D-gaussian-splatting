import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pycolmap as colmap
from gaussian3d import Gaussian3D
import numpy as np
from camera import Camera
from PIL import Image
import matplotlib.pyplot as plt
from main import render_multiple_gaussians  # render function
import torch
import torch.nn as nn
import torch.optim as optim
from gaussianmodeltrain import GaussianModel, render_gaussians_torch, train

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
        target_img = Image.open(img_path).convert("RGB")
        target_img = target_img.resize((800, 600), Image.LANCZOS) # downscale it to see if we get better results
        target_img = np.array(target_img).astype(np.float32) / 255.0

        
        camera_center = -R.T @ t  # actual camera position in world space
        # Create Camera object
        cam = Camera(
            R=R,
            T=camera_center, #changed this
            FoVx=FoVx,
            FoVy=FoVy,
            image_width=800,#width
            image_height=600, #height
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
# preview the render
def preview_pytorch_render(gaussians_list, cameras_data, cam_index=0, device='cpu'):
    model = GaussianModel(gaussians_list)
    cam_data = cameras_data[cam_index]
    camera = cam_data['camera']
    target = cam_data['target_image']

    with torch.no_grad():
        rendered = render_gaussians_torch(model, camera, device)
    rendered_np = rendered.cpu().numpy()
    loss = np.mean(np.abs(target - rendered_np))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(np.clip(target, 0, 1))
    axes[0].set_title("Target")
    axes[1].imshow(np.clip(rendered_np, 0, 1))
    axes[1].set_title(f"Rendered | L1: {loss:.4f}")
    axes[2].imshow(np.clip(np.abs(target - rendered_np), 0, 1))
    axes[2].set_title("Difference")
    for ax in axes: ax.axis('off')
    plt.tight_layout()
    plt.savefig("preview_render.png", dpi=150, bbox_inches='tight')
    plt.close()
    
# # Test it:
gaussians, recon = test_initialization()
cameras_data = load_cameras_from_colmap(recon)

# # # Run the test
# test_initial_render(gaussians, cameras_data)
# model = GaussianModel(gaussians)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
preview_pytorch_render(gaussians, cameras_data, cam_index=0, device=device)

model, losses = train(gaussians, cameras_data,  n_epochs=50, device=device)

# # plot loss curve
plt.figure()
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("L1 Loss")
plt.title("Training Loss")
plt.savefig("loss_curve.png")
plt.close()

# render final result
with torch.no_grad():
    final = render_gaussians_torch(model, cameras_data[0]['camera'])
    final_np = final.cpu().numpy()
    target = cameras_data[0]['target_image']
    
    plt.figure()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(np.clip(target, 0, 1))
    axes[0].set_title("Target")
    axes[1].imshow(np.clip(final_np, 0, 1))
    axes[1].set_title("Final Render")
    axes[2].imshow(np.clip(np.abs(target - final_np), 0, 1))
    axes[2].set_title("Difference")
    for ax in axes: ax.axis('off')
    plt.tight_layout()
    plt.savefig("final_render0.png")
    plt.close()

    
    final = render_gaussians_torch(model, cameras_data[1]['camera'])
    final_np = final.cpu().numpy()
    target = cameras_data[1]['target_image']
    
    plt.figure()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(np.clip(target, 0, 1))
    axes[0].set_title("Target")
    axes[1].imshow(np.clip(final_np, 0, 1))
    axes[1].set_title("Final Render")
    axes[2].imshow(np.clip(np.abs(target - final_np), 0, 1))
    axes[2].set_title("Difference")
    for ax in axes: ax.axis('off')
    plt.tight_layout()
    plt.savefig("final_render1.png")
    plt.close()

    final = render_gaussians_torch(model, cameras_data[2]['camera'])
    final_np = final.cpu().numpy()
    target = cameras_data[2]['target_image']
    
    plt.figure()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(np.clip(target, 0, 1))
    axes[0].set_title("Target")
    axes[1].imshow(np.clip(final_np, 0, 1))
    axes[1].set_title("Final Render")
    axes[2].imshow(np.clip(np.abs(target - final_np), 0, 1))
    axes[2].set_title("Difference")
    for ax in axes: ax.axis('off')
    plt.tight_layout()
    plt.savefig("final_render2.png")
    plt.close()

    final = render_gaussians_torch(model, cameras_data[3]['camera'])
    final_np = final.cpu().numpy()
    target = cameras_data[3]['target_image']
    
    plt.figure()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(np.clip(target, 0, 1))
    axes[0].set_title("Target")
    axes[1].imshow(np.clip(final_np, 0, 1))
    axes[1].set_title("Final Render")
    axes[2].imshow(np.clip(np.abs(target - final_np), 0, 1))
    axes[2].set_title("Difference")
    for ax in axes: ax.axis('off')
    plt.tight_layout()
    plt.savefig("final_render3.png")
    plt.close()
    
    final = render_gaussians_torch(model, cameras_data[4]['camera'])
    final_np = final.cpu().numpy()
    target = cameras_data[4]['target_image']
    
    plt.figure()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(np.clip(target, 0, 1))
    axes[0].set_title("Target")
    axes[1].imshow(np.clip(final_np, 0, 1))
    axes[1].set_title("Final Render")
    axes[2].imshow(np.clip(np.abs(target - final_np), 0, 1))
    axes[2].set_title("Difference")
    for ax in axes: ax.axis('off')
    plt.tight_layout()
    plt.savefig("final_render4.png")
    plt.close()