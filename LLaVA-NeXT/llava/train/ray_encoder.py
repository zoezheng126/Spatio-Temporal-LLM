import torch 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_rays(H, W, K, c2w, inverse_y, flip_x, flip_y, mode='center'):

    i, j = torch.meshgrid(
        torch.linspace(0, W-1, W, device=c2w.device),
        torch.linspace(0, H-1, H, device=c2w.device))  # pytorch's meshgrid has indexing='ij'
    i = i.t().float()
    j = j.t().float()
    if mode == 'lefttop':
        pass
    elif mode == 'center':
        i, j = i+0.5, j+0.5
    elif mode == 'random':
        i = i+torch.rand_like(i)
        j = j+torch.rand_like(j)
    else:
        raise NotImplementedError

    if flip_x:
        i = i.flip((1,))
    if flip_y:
        j = j.flip((0,))
    if inverse_y:
        dirs = torch.stack([(i-K[0][2])/K[0][0], (j-K[1][2])/K[1][1], torch.ones_like(i)], -1).to(c2w.dtype)
    else:
        dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1).to(c2w.dtype)

    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_o = c2w[:3,3].expand(rays_d.shape)
    return rays_o, rays_d

def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d

    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]

    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)

    return rays_o, rays_d

def get_rays_of_a_view(H, W, K, c2w, ndc, inverse_y, flip_x, flip_y, mode='center'):
    rays_o, rays_d = get_rays(H, W, K, c2w, inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y, mode=mode)
    viewdirs = rays_d / rays_d.norm(dim=-1, keepdim=True)
    if ndc:
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)
    return rays_o, rays_d, viewdirs


def invert_rigid_4x4(matrix):
    """
    Inverts a 4x4 rigid transformation matrix (rotation + translation only).
    Assumes the bottom row is [0, 0, 0, 1].

    Args:
        matrix (torch.Tensor): shape (..., 4, 4)

    Returns:
        torch.Tensor: inverse matrix of shape (..., 4, 4)
    """
    assert matrix.shape[-2:] == (4, 4), "Matrix must be 4x4"
    R = matrix[..., :3, :3]      # rotation
    t = matrix[..., :3, 3:4]     # translation (column vector)

    R_inv = R.transpose(-1, -2)  # inverse of rotation
    t_inv = -R_inv @ t           # inverse translation

    # Construct the inverse matrix
    inv = torch.eye(4, dtype=matrix.dtype, device=matrix.device).expand_as(matrix).clone()
    inv[..., :3, :3] = R_inv
    inv[..., :3, 3] = t_inv.squeeze(-1)

    return inv

def transform_to_first_camera_video(c2w_first, c2ws):
    """Transforms c2w and pcd to the first camera's coordinate system without torch.linalg.inv."""

    R = c2w_first[..., :3, :3]  # [3, 3] or [B, 3, 3]
    t = c2w_first[..., :3, 3:]  # [3, 1] or [B, 3, 1]
    
    R_inv = R.transpose(-1, -2)  # [3, 3] or [B, 3, 3]
    t_inv = -R_inv @ t  # [3, 1] or [B, 3, 1]
    c2w_first_inv = torch.eye(4, device=c2w_first.device, dtype=c2w_first.dtype).expand_as(c2w_first).clone()
    c2w_first_inv[..., :3, :3] = R_inv
    c2w_first_inv[..., :3, 3:] = t_inv

    c2ws_transformed = c2w_first_inv[:, None] @ c2ws  # If c2ws shape is [B, N, 4, 4]
    return c2ws_transformed

def transform_to_first_camera_pointcloud(pointcloud, c2ws):
    """
    Transforms pointcloud (world-coord) to the coordinate frame of the first video frame.

    Args:
        pointcloud (torch.Tensor): shape [1, N, 3], in world coordinates.
        c2ws (torch.Tensor): shape [1, T, 4, 4], camera-to-world matrices for T frames.

    Returns:
        torch.Tensor: Transformed pointcloud in first-frame coordinates, shape [1, N, 3]
    """
    assert pointcloud.shape[0] == 1, "Only supports batch size 1 for now"
    assert c2ws.shape[0] == 1, "Only supports batch size 1 for now"
    
    pc = pointcloud[0]  # [N, 3]
    c2w_0 = c2ws[0, 0]  # [4, 4]

    # Homogenize pointcloud
    ones = torch.ones((pc.shape[0], 1), dtype=pc.dtype, device=pc.device)
    pc_h = torch.cat([pc, ones], dim=1)  # [N, 4]

    # Compute world-to-camera for frame 0
    w2c_0 = invert_rigid_4x4(c2w_0)  # [4, 4]

    # Transform
    w2c_0 = w2c_0.to(torch.bfloat16)
    pc_h = pc_h.to(torch.bfloat16)
    pc_cam = (w2c_0 @ pc_h.T).T[:, :3]  # [N, 3]
    pc_cam = pc_cam.unsqueeze(0)  # back to shape [1, N, 3]

    return pc_cam

def compute_rays_for_pcd(pcd_transformed):
    """Computes rays for each point in the transformed point cloud."""
    # print(pcd_transformed.shape)
    batch_size, num_points, _ = pcd_transformed.shape
    rays_o = torch.zeros((batch_size, num_points, 3)).to(pcd_transformed.device)
    rays_d = torch.nn.functional.normalize(pcd_transformed - rays_o, dim=-1).to(pcd_transformed.device)
    viewdirs = torch.nn.functional.normalize(rays_d, dim=-1).to(pcd_transformed.device)

    return rays_o, rays_d, viewdirs

def compute_rays_embedding_for_pcd(viewdirs, view_base_pe=42, view_freq=None):
    if view_freq is None:
        view_freq = 2 ** torch.arange(0, view_base_pe)  
    view_freq = view_freq.view(1, 1, 1, -1)  # Shape: (1, 1, 1, L)
    viewdirs_emb = (viewdirs.unsqueeze(-1) * view_freq).flatten(-2)  # Shape: (batch_size, num_points, 3 × L)
    viewdirs_emb = torch.cat([viewdirs, viewdirs_emb.sin(), viewdirs_emb.cos()], -1)  # Shape: (batch_size, num_points, 3 + 2 × (3 × L))
    viewdirs_emb = torch.cat([viewdirs_emb, torch.zeros_like(viewdirs_emb[..., :1])], -1)
    return viewdirs_emb

def compute_rays_for_video(H, W, K, c2ws_transformed):
    """Computes rays for each video frame using the provided function."""
    batch_size, num_frames, _, _ = c2ws_transformed.shape
    all_rays_o, all_rays_d, all_viewdirs = [], [], []
    for b in range(batch_size):
        batch_rays_o, batch_rays_d, batch_viewdirs = [], [], []
        for f in range(num_frames):
            c2w = c2ws_transformed[b, f]
            rays_o, rays_d, viewdirs = get_rays_of_a_view(H, W, K, c2w, ndc=False, inverse_y=True, flip_x=False, flip_y=False)
            batch_rays_o.append(rays_o.view(-1, 3))
            batch_rays_d.append(rays_d.view(-1, 3))
            batch_viewdirs.append(viewdirs.view(-1, 3))

        all_rays_o = torch.stack(batch_rays_o)
        all_rays_d = torch.stack(batch_rays_d)
        all_viewdirs = torch.stack(batch_viewdirs)
        
    return all_rays_o, all_rays_d, all_viewdirs

def generate_random_sample(device='cuda:1'):
    # input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes
    sample = {
        'input_ids': torch.randint(0, 100, (1, 113), device=device),
        'position_ids': torch.randint(0, 2, (1, 113), device=device),
        'attention_mask': torch.ones((1, 113), dtype=torch.bool, device=device),
        'labels': torch.randint(-100, 100, (1, 113), device=device),
        'image_sizes': [3763584],
        'modalities': ['video'],
        'images': [torch.randn((2, 3, 384, 384), device=device)],
        'past_key_values': None,
        'c2ws': torch.stack([torch.eye(4).to(device).repeat(2, 1, 1) for _ in range(1)]),
        'K': torch.tensor([[500.0, 0, 192], [0, 500.0, 192], [0, 0, 1]]).to(device)
    }
    return sample

def compute_bbox_rays_multimodal(c2ws, pcd=None, H=None, W=None, K=None):
    """Computes rays for each batch element separately."""
    if pcd is not None: 
        pcd_transformed = transform_to_first_camera_pointcloud(pcd, c2ws)
        torch.save(pcd_transformed, 'pcd_transformed.pth')
        rays_o_pcd, rays_d_pcd, viewdirs_pcd = compute_rays_for_pcd(pcd_transformed)
        return rays_o_pcd, rays_d_pcd, viewdirs_pcd

    c2w_first = c2ws[:, 0]
    c2ws_transformed = transform_to_first_camera_video(c2w_first, c2ws)
    rays_o_vid, rays_d_vid, viewdirs_vid = compute_rays_for_video(H, W, K, c2ws_transformed)
    c2w_first = c2ws_transformed[0][0]
    rays_o_vid, rays_d_vid, viewdirs_vid = rays_o_vid.to(c2w_first.dtype), rays_d_vid.to(c2w_first.dtype), viewdirs_vid.to(c2w_first.dtype)
    rays_d_vid = torch.matmul(rays_d_vid, c2w_first[:3, :3])       # -> [B, N, 3]
    viewdirs_vid = torch.matmul(viewdirs_vid, c2w_first[:3, :3])   # -> [B, N, 3]

    return rays_o_vid, rays_d_vid, viewdirs_vid

class RayDirectionEncoderForImage(nn.Module):
    def __init__(self, num_patches=729, feature_dim=3584, view_base_pe=127, patch_size=14, modality='image', **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.view_base_pe = view_base_pe

        self.view_freq = 2 ** torch.arange(0, view_base_pe)
        self.img_ray_projection_to_qformer = nn.Linear(3 + 2 * (3 * view_base_pe), feature_dim)

    @staticmethod
    def compute_rays_for_video_batch(c2ws, H, W, K):
        """
            - rays_o_vid: [B, T, HW, 3]
            - rays_d_vid: [B, T, HW, 3]
            - viewdirs_vid: [B, T, HW, 3]
        """
        c2w_first = c2ws[:, 0]


        c2ws_transformed = transform_to_first_camera_video(c2w_first, c2ws)

        rays_o_vid, rays_d_vid, viewdirs_vid = compute_rays_for_video(H, W, K, c2ws_transformed)

        c2w_first = c2ws[0][0]  # shape: [4,4]
        rays_d_vid = torch.matmul(rays_d_vid, c2w_first[:3, :3])
        viewdirs_vid = torch.matmul(viewdirs_vid, c2w_first[:3, :3])
        rays_o_vid = rays_o_vid.to(c2w_first.dtype)
        rays_d_vid = rays_d_vid.to(c2w_first.dtype)
        viewdirs_vid = viewdirs_vid.to(c2w_first.dtype)

        return rays_o_vid, rays_d_vid, viewdirs_vid
    

    def forward(self, rays_d, H=None, W=None, is_normalized=True):
        viewdirs = rays_d if is_normalized else torch.nn.functional.normalize(rays_d, dim=-1)
        batch_size, num_frames, HW, _ = viewdirs.shape

        if H is None or W is None:
            H = W = int(HW ** 0.5)
        H_patches = W_patches = int(self.num_patches ** 0.5)

        rays_d_pooled = F.adaptive_avg_pool2d(
            viewdirs.view(batch_size * num_frames, H, W, 3).permute(0, 3, 1, 2),
            (H_patches, W_patches)
        ).permute(0, 2, 3, 1).view(batch_size, num_frames, -1, 3)

        view_freq = self.view_freq.to(rays_d.device).view(1, 1, 1, -1)
        viewdirs_emb = (rays_d_pooled.unsqueeze(-1) * view_freq).flatten(-2)
        viewdirs_emb = torch.cat([rays_d_pooled, viewdirs_emb.sin(), viewdirs_emb.cos()], -1)

        img_pos = self.img_ray_projection_to_qformer(viewdirs_emb)
        return img_pos  # Shape: (B, num_frames, num_patches, feature_dim)
    
class RayDirectionEncoderForPointCloud(nn.Module):
    def __init__(self, feature_dim=3584, n_points=1024, view_base_pe=127, aligned_method='linear', modality='pointcloud', **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_points = n_points
        self.view_base_pe = view_base_pe
        self.aligned_method = aligned_method

        self.view_freq = 2 ** torch.arange(0, view_base_pe)
        self.pcd_ray_projection_to_qformer = nn.Linear(3 + 2 * (3 * view_base_pe), feature_dim)

    @staticmethod
    def compute_rays_for_pointcloud_batch(c2ws, pcd):
        """
            - rays_o_pcd: [B, N, 3]
            - rays_d_pcd: [B, N, 3]
            - viewdirs_pcd: [B, N, 3]
        """
        pcd_transformed = transform_to_first_camera_pointcloud(pcd, c2ws)

        rays_o_pcd, rays_d_pcd, viewdirs_pcd = compute_rays_for_pcd(pcd_transformed)

        return rays_o_pcd, rays_d_pcd, viewdirs_pcd
    

    def forward(self, rays_d, is_normalized=True):
        viewdirs = rays_d if is_normalized else torch.nn.functional.normalize(rays_d, dim=-1)
        view_freq = self.view_freq.to(rays_d.device).view(1, 1, 1, -1)

        viewdirs_emb = (viewdirs.unsqueeze(-1) * view_freq).flatten(-2)
        viewdirs_emb = torch.cat([viewdirs, viewdirs_emb.sin(), viewdirs_emb.cos()], -1)

        viewdirs_emb_pooled = F.adaptive_avg_pool1d(viewdirs_emb.permute(0, 2, 1), self.n_points).permute(0, 2, 1)

        if self.aligned_method == 'linear':
            pcd_pos = self.pcd_ray_projection_to_qformer(viewdirs_emb_pooled)
        elif self.aligned_method == 'dummy':
            pcd_pos = torch.cat([viewdirs_emb_pooled, torch.zeros_like(viewdirs_emb_pooled[..., :1])], -1)

        return pcd_pos  # Shape: (B, n_points, feature_dim)





