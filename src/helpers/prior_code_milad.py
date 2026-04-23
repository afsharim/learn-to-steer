########################################################
# ==========================================
# NOVEL ANGULAR STEERING ADDITIONS
# ==========================================
global ANGULAR_COMPASS
global PREDICTED_THETA
ANGULAR_COMPASS = None
PREDICTED_THETA = None

def set_angular_steering(compass_vector: torch.Tensor = None, theta: torch.Tensor = None):
    global ANGULAR_COMPASS, PREDICTED_THETA
    if compass_vector is not None:
        ANGULAR_COMPASS = compass_vector
    if theta is not None:
        PREDICTED_THETA = theta

def apply_angular_steering(
    x: torch.Tensor,
    compass_vector: torch.Tensor,
    theta: float,
    only_generated_tokens: bool = False,
    include_last_prompt_token: bool = False,
    start_prompt_token_idx: int = 0,
) -> torch.Tensor:
    if compass_vector is None:
        return x
        
    # --- BULLETPROOF SHAPE FIX ---
    # No matter if compass_vector is [4096], [6300, 4096], or [1, 6300, 4096]
    # This reshapes it to [N, 4096] and grabs the very first vector, 
    # guaranteeing a strict 1D tensor of shape [4096].
    z = compass_vector.to(x.device).to(x.dtype)
    z = z.reshape(-1, x.shape[-1])[0] 
    
    # Ensure theta is treated as a float (in case bash passes it as a string)
    theta_val = float(theta)
    # -----------------------------
    
    # Sequence generation (first pass)
    if x.shape[1] > 1:
        if only_generated_tokens:
            return x
        if include_last_prompt_token:
            start_prompt_token_idx = -1
            
        if start_prompt_token_idx > 0 or start_prompt_token_idx == -1:
            x_ = x[:, start_prompt_token_idx:, :]
            
            # --- Gram-Schmidt Orthogonalization & Rotation ---
            theta_t = torch.tensor(theta_val, device=x_.device, dtype=x_.dtype)
            
            h_norm = torch.norm(x_, dim=-1, keepdim=True)
            e1 = x_ / (h_norm + 1e-8)
            
            projection = torch.sum(z * e1, dim=-1, keepdim=True) * e1
            z_perp = z - projection
            e2 = z_perp / (torch.norm(z_perp, dim=-1, keepdim=True) + 1e-8)
            
            cos_theta = torch.cos(theta_t)
            sin_theta = torch.sin(theta_t)
            
            x_rotated = h_norm * (cos_theta * e1 + sin_theta * e2)
            x[:, start_prompt_token_idx:, :] = x_rotated
            return x
            
    # Autoregressive decoding (token-by-token pass)
    theta_t = torch.tensor(theta_val, device=x.device, dtype=x.dtype)
    
    h_norm = torch.norm(x, dim=-1, keepdim=True)
    e1 = x / (h_norm + 1e-8)
    
    projection = torch.sum(z * e1, dim=-1, keepdim=True) * e1
    z_perp = z - projection
    e2 = z_perp / (torch.norm(z_perp, dim=-1, keepdim=True) + 1e-8)
    
    cos_theta = torch.cos(theta_t)
    sin_theta = torch.sin(theta_t)
    
    x = h_norm * (cos_theta * e1 + sin_theta * e2)
    return x
            
# ==========================================
########################################################
