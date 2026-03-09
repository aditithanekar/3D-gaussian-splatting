import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import numpy as np
import math
import matplotlib.pyplot as plt


#  Revised Differentiable Gaussian model from train.py for debugging

class GaussianModel(nn.Module):
    """
    Holds all Gaussian parameters as learnable tensors.
    Initialize from your existing list of Gaussian3D objects.
    """
    def __init__(self, gaussians):
        super().__init__()
        centers    = np.stack([g.center   for g in gaussians])   # (N,3)
        scales     = np.stack([g.scale    for g in gaussians])   # (N,3)
        rotations  = np.stack([g.rotation for g in gaussians])   # (N,4)  quaternions
        colors     = np.stack([g.color    for g in gaussians])   # (N,3)
        opacities  = np.array([g.alpha    for g in gaussians])   # (N,)

        self.centers   = nn.Parameter(torch.tensor(centers,   dtype=torch.float32))
        self.scales    = nn.Parameter(torch.tensor(scales,    dtype=torch.float32))
        self.rotations = nn.Parameter(torch.tensor(rotations, dtype=torch.float32))
        self.colors    = nn.Parameter(torch.tensor(colors,    dtype=torch.float32))
        # sigmoid-space so alpha stays in (0,1) during optimization
        inv_sig        = np.log(opacities / (1.0 - np.clip(opacities, 1e-6, 1-1e-6)))
        self.raw_alpha = nn.Parameter(torch.tensor(inv_sig,   dtype=torch.float32))

    @property
    def alphas(self):
        return torch.sigmoid(self.raw_alpha)   # (N,)


#  Camera helper (no grad needed here)
def camera_matrices(camera):
    """Return R, T as float32 tensors (not parameters)."""
    R = torch.tensor(np.array(camera.R), dtype=torch.float32)  # (3,3)
    T = torch.tensor(np.array(camera.T), dtype=torch.float32)  # (3,)
    fx = camera.image_width  / (2 * math.tan(camera.FoVx / 2))
    fy = camera.image_height / (2 * math.tan(camera.FoVy / 2))
    cx = camera.image_width  / 2.0
    cy = camera.image_height / 2.0
    return R, T, fx, fy, cx, cy


#  Covariance from scale + quaternion
def build_covariance_3d(scales, quats):
    """
    scales : (N,3)
    quats  : (N,4)  [qr, qi, qj, qk]
    returns: (N,3,3) covariance matrices  Sigma = R S S^T R^T
    """
    N = scales.shape[0]

    # Normalize quaternions
    q = quats / (quats.norm(dim=1, keepdim=True) + 1e-8)  # (N,4)
    qr, qi, qj, qk = q[:,0], q[:,1], q[:,2], q[:,3]

    # Build rotation matrices (N,3,3)
    R = torch.stack([
        1-2*(qj*qj+qk*qk),  2*(qi*qj-qr*qk),   2*(qi*qk+qr*qj),
        2*(qi*qj+qr*qk),    1-2*(qi*qi+qk*qk),  2*(qj*qk-qr*qi),
        2*(qi*qk-qr*qj),    2*(qj*qk+qr*qi),    1-2*(qi*qi+qj*qj)
    ], dim=1).reshape(N, 3, 3)

    # Scale matrices (N,3,3)  — diagonal
    S = torch.diag_embed(scales)   # (N,3,3)

    RS = R @ S
    return RS @ RS.transpose(1, 2)   # (N,3,3)


#  Project covariance to 2-D screen space
def project_covariance_2d(Sigma3d, view_points, R_cam, fx, fy):
    """
    Sigma3d   : (N,3,3)
    view_points: (N,3)   points already in camera space
    R_cam     : (3,3)
    Returns   : (N,2,2) screen-space covariance
    """
    x, y, z = view_points[:,0], view_points[:,1], view_points[:,2]

    # Jacobian of perspective projection  (N,2,3)
    zero = torch.zeros_like(x)
    J = torch.stack([
        fx/z,    zero,    -fx*x/(z*z),
        zero,    fy/z,    -fy*y/(z*z)
    ], dim=1).reshape(-1, 2, 3)

    # W = R_cam  (same for all gaussians)
    W = R_cam.unsqueeze(0).expand(len(view_points), -1, -1)   # (N,3,3)

    JW = J @ W                         # (N,2,3)
    cov2d = JW @ Sigma3d @ JW.transpose(1, 2)   # (N,2,2)

    # Small regularization so cov is always invertible
    eye2 = torch.eye(2, device=cov2d.device).unsqueeze(0)
    cov2d = cov2d + eye2 * 1e-4

    return cov2d   # (N,2,2)


#  Main differentiable renderer
def render_gaussians_torch(model: GaussianModel, camera, device='cpu'):

    #Fully differentiable renderer — gradients flow to all GaussianModel params.
    #Returns image tensor  (H, W, 3)  on device, values in [0,1].
    H, W = camera.image_height, camera.image_width
    R_cam, T_cam, fx, fy, cx, cy = camera_matrices(camera)
    R_cam = R_cam.to(device)
    T_cam = T_cam.to(device)

    centers   = model.centers.to(device)    # (N,3)
    
    scales    = model.scales.to(device)     # (N,3)
    rotations = model.rotations.to(device)  # (N,4)
    colors    = torch.sigmoid(model.colors).to(device)  # (N,3)  keep in [0,1]
    alphas    = model.alphas.to(device)                  # (N,)

    #  World to camera coords 
    view_points = (centers - T_cam[None,:]) @ R_cam.T   # (N,3)
    depths = view_points[:, 2]                   # (N,)

    # Filter behind-camera gaussians (keep as mask, don't break graph)
    valid = depths > 0.0
    if not valid.any():
        return torch.zeros(3, H, W, device=device)
    
    print(f"centers shape: {centers.shape}")
    print(f"depths min/max: {depths.min():.3f} / {depths.max():.3f}")
    print(f"any nan in centers: {torch.isnan(centers).any()}")
    print(f"any nan in depths: {torch.isnan(depths).any()}")
    print(f"valid count: {valid.sum()} / {len(valid)}")

    view_points = view_points[valid]
    colors_v   = colors[valid]
    alphas_v   = alphas[valid]
    scales_v   = scales[valid]
    rots_v     = rotations[valid]
    depths_v   = depths[valid]

    #  Project centers to pixel coords 
    x, y, z = view_points[:,0], view_points[:,1], view_points[:,2]
    x_normalized = x / z
    y_normalized = y / z 
    px = x_normalized * fx + cx   # (N,)
    py = y_normalized * fy + cy   # (N,)
    


    #  Build 2-D covariance 
    Sigma3d = build_covariance_3d(scales_v, rots_v)                     # (N,3,3)
    cov2d   = project_covariance_2d(Sigma3d, view_points, R_cam, fx, fy) # (N,2,2)

    # Invert 2x2 analytically (faster + stable)
    a = cov2d[:,0,0]; b = cov2d[:,0,1]; d = cov2d[:,1,1]
    det = a*d - b*b + 1e-8
    inv_a =  d / det
    inv_b = -b / det
    inv_d =  a / det   # shape (N,)

    # Depth sorting
    order = depths_v.argsort(descending=True)   # farthest first
    px, py       = px[order],      py[order]
    inv_a, inv_b, inv_d = inv_a[order], inv_b[order], inv_d[order]
    colors_v     = colors_v[order]
    alphas_v     = alphas_v[order]
    scales_v     = scales_v[order]

    # Rasterize — vectorized over pixels 
    # Pixel grid
    ys = torch.arange(H, dtype=torch.float32, device=device)  # (H,)
    xs = torch.arange(W, dtype=torch.float32, device=device)  # (W,)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')    # (H,W) each

    image         = torch.zeros(H, W, 3, device=device)
    transmittance = torch.ones( H, W,    device=device)

    N = px.shape[0]
    # Process in chunks to avoid huge (N,H,W) tensors on GPU memory
    chunk = 8

    for start in range(0, N, chunk):
        end = min(start + chunk, N)

        # offsets from each gaussian center to every pixel  (C, H, W)
        C = end - start
        dx = grid_x.unsqueeze(0) - px[start:end].reshape(C, 1, 1)   # (C,H,W)
        dy = grid_y.unsqueeze(0) - py[start:end].reshape(C, 1, 1)   # (C,H,W)

        ia = inv_a[start:end].reshape(C, 1, 1)
        ib = inv_b[start:end].reshape(C, 1, 1)
        id_ = inv_d[start:end].reshape(C, 1, 1)

        # Mahalanobis distance
        maha = 0.5 * (dx*dx*ia + 2*dx*dy*ib + dy*dy*id_)   # (C,H,W)

        # Bounding-box mask: skip pixels more than 3.5σ away (saves compute)
        sigma_x = torch.sqrt(1.0 / (ia.squeeze() + 1e-8))
        sigma_y = torch.sqrt(1.0 / (id_.squeeze() + 1e-8))
        radius = torch.max(sigma_x, sigma_y).reshape(C, 1, 1) * 3.5
        in_bounds = (dx.abs() < radius) & (dy.abs() < radius)

        gauss_w = torch.exp(-maha) * in_bounds.float()   # (C,H,W)

        a_chunk = alphas_v[start:end].reshape(C, 1, 1)   # (C,1,1)
        c_chunk = colors_v[start:end]                     # (C,3)

        for i in range(C):
            alpha_map = a_chunk[i, 0, 0] * gauss_w[i]       # (H,W)
            # alpha composite
            contrib = transmittance * alpha_map              # (H,W)
            image += contrib.unsqueeze(-1) * c_chunk[i]     # (H,W,3)
            transmittance = transmittance * (1.0 - alpha_map)

    return image   # (H,W,3)


#  Training loop
def train(gaussians_list, cameras_data, n_epochs=500, lr=1e-3, device='cpu'):
    """
    gaussians_list : list of Gaussian3D (your existing objects)
    cameras_data   : list of dicts with keys 'camera' and 'target_image' (H,W,3 float32 numpy)
    """
    torch.autograd.set_detect_anomaly(True)
    import psutil, gc
    model = GaussianModel(gaussians_list).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    l1_loss = nn.L1Loss()
    loss_history = []


    for epoch in range(n_epochs):
        print(f"\n--- Iter {epoch}/{n_epochs} ---", flush=True)
        total_loss = 0.0
        mem = psutil.virtual_memory()
        print(f"RAM: {mem.used/1e9:.1f}GB / {mem.total/1e9:.1f}GB", flush=True)
        if torch.cuda.is_available():
            print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB allocated", flush=True)
            torch.cuda.empty_cache()
        #manual garbage collection
        gc.collect()
        for cam_data in cameras_data:
            camera       = cam_data['camera']
            target_np    = cam_data['target_image']   # (H,W,3) float32

            target = torch.tensor(target_np, dtype=torch.float32, device=device)

            optimizer.zero_grad()

            rendered = render_gaussians_torch(model, camera, device=device)

            # Resize target if needed (COLMAP images may differ slightly)
            if rendered.shape != target.shape:
                # simple nearest-neighbor crop/pad — or use torchvision.transforms
                h = min(rendered.shape[0], target.shape[0])
                w = min(rendered.shape[1], target.shape[1])
                rendered = rendered[:h, :w]
                target   = target[:h, :w]

            loss = l1_loss(rendered, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / max(len(cameras_data), 1)
        loss_history.append(avg)


        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"Epoch {epoch:4d} | loss {avg:.5f}")

    return model, loss_history



