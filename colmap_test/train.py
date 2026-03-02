import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
from testcolmap import test_initialization, load_cameras_from_colmap
import torch.optim as optim
from main import render_multiple_gaussians
from gaussian3d import Gaussian3D
import matplotlib.pyplot as plt
from render import render_differentiable

class GaussianModel(nn.Module):
    # Optimizable Gaussian parameters
    
    def __init__(self, initial_gaussians):
        super().__init__()
        
        N = len(initial_gaussians)
        
        # get parameters all in an np array together from Gaussian3D objects
        positions = np.array([g.center for g in initial_gaussians])
        scales = np.array([g.scale for g in initial_gaussians])
        rotations = np.array([g.rotation for g in initial_gaussians])
        colors = np.array([g.color for g in initial_gaussians])
        opacities = np.array([[g.alpha] for g in initial_gaussians])
        
        # turn np arrays into tensor parameters to be trainable
        self.positions = nn.Parameter(torch.from_numpy(positions).float())
        self.scales = nn.Parameter(torch.from_numpy(scales).float())
        self.rotations = nn.Parameter(torch.from_numpy(rotations).float())
        self.colors = nn.Parameter(torch.from_numpy(colors).float())
        self.opacities = nn.Parameter(torch.from_numpy(opacities).float())

        # clamp initial scales to prevent exploding bounding boxes
        with torch.no_grad():
            self.scales.clamp_(-3, 3)
            self.scales.fill_(1.0)  # exp(1) ≈ 2.7 world units, much more visible
        
        print(f"Initialized model with {N} Gaussians")
        print(f"Total parameters: {sum(p.numel() for p in self.parameters())}")
    # FOR TRAINING 
    def get_gaussian_tensors(self):
        # Return raw tensors without numpy — for training 
        # apply activations —  like the  3DGS repo gaussian_model
        scales     = torch.exp(self.scales)               # → positive scales
        opacities  = torch.sigmoid(self.opacities)        #  [0, 1]
        rotations  = torch.nn.functional.normalize(self.rotations, dim=-1)  # unit quaternions

        
        return self.positions, scales, rotations, self.colors, opacities
    
    # FOR VISUALIZATION ONLY
    @torch.no_grad()
    def get_gaussians(self):
        # Returns list of Gaussian3D objects — only call when you want to visualize
        gaussians = []
        N = self.positions.shape[0]
        
        # Apply the same activations as in training
        scales_act     = torch.exp(self.scales).cpu().numpy()     # positive
        opacities_act  = torch.sigmoid(self.opacities).cpu().numpy()
        rotations_act  = torch.nn.functional.normalize(self.rotations, dim=-1).cpu().numpy()
        colors_act     = torch.sigmoid(self.colors).cpu().numpy()   # or clamp
        
        for i in range(N):
            g = Gaussian3D(
                center    = self.positions[i].cpu().numpy(),
                scale     = scales_act[i],                     # already exp'ed
                rotation  = rotations_act[i],
                color     = np.clip(colors_act[i], 0, 1),      # safety clip
                opacity   = float(opacities_act[i])            # already sigmoid
            )
            gaussians.append(g)
        
        return gaussians


def train_simple(model, cameras_data, num_iterations=50, lr=0.001):
    # Simple CPU training loop
    
    # Optimizer 
    optimizer = optim.Adam([
        {'params': [model.positions], 'lr': lr * 1.0},
        {'params': [model.scales], 'lr': lr * 0.5},
        {'params': [model.rotations], 'lr': lr * 0.1},
        {'params': [model.colors], 'lr': lr * 1.0},
        {'params': [model.opacities], 'lr': lr * 0.5},
    ])
    
    losses = []
    
    # Downscale for speed
    scale = 0.25
    
    for iteration in range(num_iterations):
        # Pick random camera
        cam_idx = np.random.randint(len(cameras_data))
        cam_data = cameras_data[cam_idx]
        camera = cam_data['camera']
        target = cam_data['target_image']
        
        # Downscale target
        small_w = int(camera.image_width * scale)
        small_h = int(camera.image_height * scale)
        
        from PIL import Image as PILImage
        target_pil = PILImage.fromarray((target * 255).astype(np.uint8))
        target_small = np.array(target_pil.resize((small_w, small_h))) / 255.0
        
        # Get current Gaussians
        current_gaussians = model.get_gaussians()
        
        # Render
        rendered = render_multiple_gaussians(
            current_gaussians, camera, small_w, small_h
        )
        
        # Convert to torch
        rendered_torch = torch.from_numpy(rendered).float()
        target_torch = torch.from_numpy(target_small).float()
        
        # Loss
        loss = torch.mean(torch.abs(rendered_torch - target_torch))
        
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        print(f"Iter {iteration+1}/{num_iterations}: Loss = {loss.item():.4f}, Camera = {cam_data['name']}")
    
    return losses

# more minimal version first to ensure it's working. later todos: densification, and more
def train_gaussians(model, cameras_data, num_iterations=10, lr=0.002):
    torch.autograd.set_detect_anomaly(True)
    import psutil, gc
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for i in range(1, num_iterations + 1):
        # printing and RAM check for memory
        print(f"\n--- Iter {i}/{num_iterations} ---", flush=True)
    
        mem = psutil.virtual_memory()
        print(f"RAM: {mem.used/1e9:.1f}GB / {mem.total/1e9:.1f}GB", flush=True)
        if torch.cuda.is_available():
            print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB allocated", flush=True)
            torch.cuda.empty_cache()
        #manual garbage collection
        gc.collect()
        #training code
        optimizer.zero_grad()
        
        cam = cameras_data[np.random.randint(len(cameras_data))]
        gt = torch.from_numpy(cam['target_image']).float().permute(2, 0, 1).to(device)  # (3,H,W)
        
        pos, sca, rot, col, opa = model.get_gaussian_tensors()
        rendered = render_differentiable(pos, sca, rot, col, opa, cam['camera'], gt.shape[2], gt.shape[1], device)
        
        loss = torch.mean(torch.abs(rendered - gt))
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        if i % 10 == 0:
            print(f"Iter {i:3d}  loss = {loss.item():.5f}")
    
    return losses



# Test it
print("Loading initial Gaussians...")
gaussians, recon = test_initialization()
cameras_data = load_cameras_from_colmap(recon)

print("\nCreating PyTorch model...")
model = GaussianModel(gaussians)

print("\n✓ Model ready for training!")
print(f"  Positions: {model.positions.shape}")
print(f"  Scales: {model.scales.shape}")
print(f"  Colors: {model.colors.shape}")


# Run training
print("STARTING TRAINING")
losses = train_gaussians(model, cameras_data, num_iterations=300, lr=0.001)


# After training
print("\nFinal comparison render...")
from google.colab import files
with torch.no_grad():
    cam_data = cameras_data[0]  # pick any view, preferably one not overfitted
    camera = cam_data['camera']
    gt_img = cam_data['target_image']  # numpy (H,W,3)
    
    h, w = gt_img.shape[:2]
    render_size_factor = 0.5
    render_h = int(h * render_size_factor)
    render_w = int(w * render_size_factor)
    
    pos, sca, rot, col, opa = model.get_gaussian_tensors()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rendered = render_differentiable(pos, sca, rot, col, opa, camera, render_w, render_h, device)
    rendered_np = rendered.detach().cpu().permute(1, 2, 0).clamp(0,1).numpy()
    
    # show side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(gt_img)
    axes[0].set_title("Ground Truth")
    axes[0].axis('off')
    
    axes[1].imshow(rendered_np)
    axes[1].set_title("Model render after training")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig("final_comparison.png")
    files.download("final_comparison.png")
    print("hewwo you downloaded the final render")

    # plt.show()

# Plot
plt.plot(losses)
plt.xlabel("Iteration")
plt.ylabel("L1 Loss")
plt.title("Training Progress")
plt.grid(True)
plt.savefig("loss_curve.png")
files.download("loss_curve.png")
print("hewwo you downloaded the loss")
# plt.show()


