import torch
import torch.nn as nn
import torch.optim as optim
import math


#pytorch implemenation of render_multiple_gaussians in main.py

def render_differentiable(
    positions: torch.Tensor,    # (N, 3)
    scales: torch.Tensor,       # (N, 3)
    rotations: torch.Tensor,    # (N, 4) quats
    colors: torch.Tensor,       # (N, 3)
    opacities: torch.Tensor,    # (N, 1)
    camera,                     # your camera object with intrinsics, extrinsics
    width: int,
    height: int,
    device: str = 'cpu'
) -> torch.Tensor:
    #Differentiable forward render → returns (3, H, W)
    
    # Move everything to device first in case using colab
    positions = positions.to(device)
    scales    = scales.to(device)
    rotations = rotations.to(device)
    colors    = colors.to(device)
    opacities = opacities.to(device)
    
    # Ensure camera extrinsics are torch tensors without gradients
    if not isinstance(camera.T, torch.Tensor):
        camera.T = torch.tensor(camera.T, dtype=torch.float32, device=device)
    else:
        camera.T = camera.T.to(device).detach()   # remove grad if it had any

    if not isinstance(camera.R, torch.Tensor):
        camera.R = torch.tensor(camera.R, dtype=torch.float32, device=device)
    else:
        camera.R = camera.R.to(device).detach()
    
    image = torch.zeros((height, width, 3), dtype=torch.float32, device=device)
    transmittance = torch.ones((height, width), dtype=torch.float32, device=device) #initialize transmittance
    
    cam_offset_point = positions - camera.T[None, :] # allow the N,3 diff dim subtract

    view_points = (positions - camera.T[None,:]) @ camera.R.T 
    
    depths = view_points[:,2]  # depth is the z component of this 3d camera space coord

    # filter out Gaussians behind camera
    valid = depths > 0   #  avoids div by 0 errors too
    if not valid.any():
        return torch.zeros(3, height, width, device=device)

    positions = positions[valid]
    view_points = view_points[valid]
    depths = depths[valid]
    scales = scales[valid]
    rotations = rotations[valid]
    colors = colors[valid]
    opacities = opacities[valid]
    
    # alpha blending must be done farthest to nearest
    sort_indices = torch.argsort(depths, descending=True)  # largest depth first
    depths = depths[sort_indices]
    view_points = view_points[sort_indices]
    scales = scales[sort_indices]
    rotations = rotations[sort_indices]
    colors = colors[sort_indices]
    opacities = opacities[sort_indices]
    
    x,y,z = view_points[:,0], view_points[:,1], view_points[:,2]

    x_normalized = x / z
    y_normalized = y / z 
    
    # reference on Field of View formula: https://www.youtube.com/watch?v=pUuAx_zFnEk
    fx = camera.image_width / (2 * math.tan(camera.FoVx / 2))
    fy = camera.image_height / (2 * math.tan(camera.FoVy / 2))
    image_center_x = camera.image_width / 2
    image_center_y = camera.image_height / 2
    
    #scale the normalized x and y to 2d, and make it so center is in top left corner to avoid negatives
    pixel_x = x_normalized * fx + image_center_x
    pixel_y = y_normalized * fy  + image_center_y
    
    
    # Jacobian 
    Jacobian = torch.zeros(len(z), 2, 3, device=device)
    Jacobian[:, 0, 0] = fx / z
    Jacobian[:, 0, 2] = -fx * x / (z ** 2)
    Jacobian[:, 1, 1] = fy / z
    Jacobian[:, 1, 2] = -fy * y / (z ** 2)
    
    W = camera.R  #rotation defines the viewing transformation
        
    
    scale_matrix = torch.diag_embed(scales) # turning 3d vec scale into 3d matrix diagonal scale matrix
    
    #normalize the quaternion rotation 
    quat_magnitudes = torch.norm(rotations, dim=1, keepdim=True)   # shape (N,1)
    # Avoid division by zero / very small values
    quat_magnitudes = torch.clamp(quat_magnitudes, min=1e-6)
        
    # we need to make sure that q(rotation) is normalized 
    normalized_rotation = rotations / quat_magnitudes  #each element divided by the magnitude
    q_r, q_i, q_j, q_k = normalized_rotation[:,0], normalized_rotation[:,1], normalized_rotation[:,2], normalized_rotation[:,3]
    rot_matrix = torch.zeros(len(q_r), 3, 3, device=device)
    
    # convert normalized_rotation to be a matrix instead of 4d vector
    # can see Equation 10 on page 13 of the paper 
    # First row of the rotation matrix
    rot_matrix[:, 0, 0] = 1 - (2 * (q_j * q_j + q_k * q_k)) 
    rot_matrix[:, 0, 1] = 2 * (q_i * q_j - q_r * q_k)
    rot_matrix[:, 0, 2] = 2 * (q_i * q_k + q_r * q_j)
    
    # Second row of the rotation matrix
    rot_matrix[:, 1, 0] = 2 * (q_i * q_j + q_r * q_k)
    rot_matrix[:, 1, 1] = 1 - (2 * (q_i * q_i + q_k * q_k))
    rot_matrix[:, 1, 2] = 2 * (q_j * q_k - q_r * q_i)
    
    # Third row of the rotation matrix
    rot_matrix[:, 2, 0] = 2 * (q_i * q_k - q_r * q_j)
    rot_matrix[:, 2, 1] = 2 * (q_j * q_k + q_r * q_i)
    rot_matrix[:, 2, 2] = 1 - (2 * (q_i * q_i + q_j * q_j))
    
    # if R is rotation matrix and S is scaling matrix, covariance Sigma = R * S * S^T * R^T  
    Sigma = rot_matrix @ scale_matrix @ scale_matrix.transpose(-1,-2) @ rot_matrix.transpose(-1, -2)
    
    covariance_camera = Jacobian @ W @ Sigma @ W.transpose(-1,-2) @ Jacobian.transpose(-1,-2)
    
    #bounding box loop
    for i in range(len(pixel_x)):
        px = pixel_x[i].item()
        py = pixel_y[i].item()

        if not (0 <= px < width and 0 <= py < height): # check pixel within bounds 
            continue

        cov_2d = covariance_camera[i]                     # (2, 2)
        cov_2d.diagonal(dim1=-2, dim2=-1).add_(1e-4)

        try:
            cov_inv = torch.inverse(cov_2d)
        except:
            continue

        # axis-aligned bounding box
        sigma_x = torch.sqrt(cov_2d[0, 0])
        sigma_y = torch.sqrt(cov_2d[1, 1])
        radius = max(sigma_x, sigma_y) * 3.5

        x_min = max(0, int(px - radius))
        x_max = min(width - 1, int(px + radius + 0.999))
        y_min = max(0, int(py - radius))
        y_max = min(height - 1, int(py + radius + 0.999))

        if x_min > x_max or y_min > y_max:
            continue

        # Grid inside bounding box
        yy, xx = torch.meshgrid(
            torch.arange(y_min, y_max + 1, device=device),
            torch.arange(x_min, x_max + 1, device=device),
            indexing='ij'
        )
        offset = torch.stack([xx - px, yy - py], dim=-1).float()   # get offset from pixel to bounding box (h', w', 2)
        dist = torch.sum(offset * (offset @ cov_inv), dim=-1)
        gaussian_weight_at_point = torch.exp(-0.5 * dist)

        alpha = opacities[i] * gaussian_weight_at_point
        
        # alpha blending:
        
        slice_y = slice(y_min, y_max + 1)
        slice_x = slice(x_min, x_max + 1)
        contrib = colors[i] * alpha[..., None]       # broadcasts to (h',w',3)

        image[slice_y, slice_x] += transmittance[slice_y, slice_x][..., None] * contrib
        transmittance[slice_y, slice_x] *= (1.0 - alpha)
        
    return image.permute(2, 0, 1).clamp(0, 1) 
    
