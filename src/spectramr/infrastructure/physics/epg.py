import torch


def simulate_differentiable_epg_fse(
    rho: torch.Tensor,
    t1: torch.Tensor,
    t2: torch.Tensor,
    excitation_fa: torch.Tensor,
    refocusing_fa: torch.Tensor,
    echo_train_length: int = 16,
    te_spacing: float | torch.Tensor = 10.0,
) -> torch.Tensor:
    """
    Simulates transverse magnetization via Differentiable Extended Phase Graph (EPG)
    for a Fast Spin Echo (FSE/TSE) sequence.

    Args:
        rho: Proton density [B, 1, H, W]
        t1: T1 relaxation time in ms [B, 1, H, W]
        t2: T2 relaxation time in ms [B, 1, H, W]
        excitation_fa: Excitation flip angle in degrees [B, 1, H, W] or scalar
        refocusing_fa: Refocusing flip angle in degrees [B, 1, H, W] or scalar
        echo_train_length: Number of echoes in the train (ETL)
        te_spacing: Time between echoes in ms (Echo Spacing); a ``[B, 1, 1, 1]`` tensor
            gives each row its own spacing, broadcast against the tissue maps.

    Returns:
        signal: Generated magnitude signal at the effective TE.
                For simplicity, returns the signal at the center of k-space
                or an averaged echo train signal depending on the contrast weighting.
    """
    # Convert parameters
    device = rho.device
    if rho.dim() != 4:
        raise ValueError(
            f"simulate_differentiable_epg_fse expects 4-D rho [B, 1, H, W]; "
            f"got rank {rho.dim()} (shape {tuple(rho.shape)}). A lower-rank rho "
            f"would broadcast silently against the 4-D flip-angle/relaxation "
            f"maps and entangle the batch/channel axes."
        )
    alpha_ex = excitation_fa * torch.pi / 180.0
    alpha_ref = refocusing_fa * torch.pi / 180.0

    # Number of states in EPG
    N = echo_train_length

    # Initialize state matrices
    # EPG states: F = F+, F* = F-, Z = Z
    # We maintain 3 x (2N+1) states per voxel, but practically we only need F0 to generate signal.
    # To keep this differentiable and memory efficient, we implement a simplified matrix transition.

    # Relaxation matrix precomputation
    E1 = torch.exp(-te_spacing / t1)
    E2 = torch.exp(-te_spacing / t2)

    # Initialize states: [F_states, F_star_states, Z_states]
    # For a full differentiable EPG over all voxels, representing all states explicitly
    # can be memory intensive. We compute the F0 state dynamically.

    # Initial excitation
    # [F0, F0*, Z0]^T = [sin(alpha_ex), 0, cos(alpha_ex)]^T * rho
    F0 = torch.sin(alpha_ex)
    F0_star = torch.zeros_like(F0)
    Z0 = torch.cos(alpha_ex)

    # Signal accumulation (e.g., center of k-space could be approx middle of echo train)
    # Here we simplify the differentiable EPG to track the primary echo pathway and
    # dominant stimulated echoes for the effective echo (e.g., echo N//2 for T2w)

    # Track only a small number of configuration states (k=0, 1, 2) for performance
    # Since D=16 (ETL), we can iteratively apply the RF mixing and relaxation/shift operators

    # This is a highly simplified differentiable proxy for the full EPG meant for deep learning
    signal = torch.zeros_like(rho)
    current_F = F0
    current_Z = Z0

    for i in range(1, echo_train_length + 1):
        # 1. Relaxation (for half echo spacing t = TE/2)
        E1_half = torch.exp(-(te_spacing / 2.0) / t1)
        E2_half = torch.exp(-(te_spacing / 2.0) / t2)

        current_F = current_F * E2_half
        current_Z = current_Z * E1_half + (1.0 - E1_half)

        # 2. RF pulse mixing (Refocusing)
        # Simplified: F+ -> sin^2(a/2) F-* + cos^2(a/2) F+ + sin(a) Z
        # Z -> -1/2 sin(a) F+ - 1/2 sin(a) F-* + cos(a) Z
        sin_sq_alpha_2 = torch.sin(alpha_ref / 2.0) ** 2
        cos_sq_alpha_2 = torch.cos(alpha_ref / 2.0) ** 2
        sin_alpha = torch.sin(alpha_ref)
        cos_alpha = torch.cos(alpha_ref)

        # To keep it completely standard without exploding state vectors,
        # we model the dominant coherence pathway (CPMG condition)
        # where F = F* = current_F (real states)
        next_F = current_F * (sin_sq_alpha_2 + cos_sq_alpha_2) + current_Z * sin_alpha
        next_Z = -current_F * sin_alpha + current_Z * cos_alpha

        current_F = next_F
        current_Z = next_Z

        # 3. Relaxation (for half echo spacing t = TE/2)
        current_F = current_F * E2_half
        current_Z = current_Z * E1_half + (1.0 - E1_half)

        # Accumulate signal (usually k-space center is at Effective TE)
        # max(1, ...) guarantees the effective-echo capture fires even for
        # ETL == 1 (where ETL//2 == 0 is never reached by the 1..ETL loop and
        # would previously fall through to the .sum() host-sync fallback).
        if i == max(1, echo_train_length // 2):
            signal = torch.abs(current_F) * rho

    # `signal` is always assigned above (the capture index is always hit), so
    # the data-dependent `signal.sum() > 0` branch — a GPU host-sync inside a
    # training-strategy forward (performance.md) — is removed.
    return signal


def simulate_full_epg_fse(
    rho: torch.Tensor,
    t1: torch.Tensor,
    t2: torch.Tensor,
    b1_plus: torch.Tensor,
    excitation_fa: float = 90.0,
    refocusing_fa: float = 180.0,
    echo_train_length: int = 16,
    te_spacing: float = 10.0,
    num_states: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Simulates a full Extended Phase Graph (EPG) with stimulated echoes and B1+ support.

    Unlike the simplified EPG which only tracks the primary pathway (CPMG), this
    maintains a full state matrix (F+, F-, Z) representing multiple coherence
    dephasing states.

    Args:
        rho: Proton density [B, 1, H, W]
        t1: T1 relaxation time in ms [B, 1, H, W]
        t2: T2 relaxation time in ms [B, 1, H, W]
        b1_plus: Transmit field inhomogeneity factor [B, 1, H, W] (ideal is 1.0)
        excitation_fa: Base excitation flip angle in degrees
        refocusing_fa: Base refocusing flip angle in degrees
        echo_train_length: Number of echoes in train
        te_spacing: Echo spacing in ms
        num_states: Number of dephasing states to track. Default is ETL.

    Returns:
        tuple containing:
        - signal: Complex transverse magnetization at echo times [B, ETL, H, W]
        - R1: Relaxation rate (1/T1) parameter map
        - R2: Relaxation rate (1/T2) parameter map
        - B1+: Transmit field map
    """
    device = rho.device
    B, C, H, W = rho.shape
    if C != 1:
        raise ValueError(
            f"simulate_full_epg_fse expects single-channel rho [B, 1, H, W]; "
            f"got channel dim C={C}. Per-echo signals are stacked along dim=1, "
            f"so C>1 would entangle the echo axis with the channel axis."
        )

    if num_states is None:
        num_states = echo_train_length + 1

    alpha_ex = b1_plus * (excitation_fa * torch.pi / 180.0)
    alpha_ref = b1_plus * (refocusing_fa * torch.pi / 180.0)

    # Pre-compute mixing matrices per voxel
    cos_a = torch.cos(alpha_ref)
    sin_a = torch.sin(alpha_ref)
    sin_sq_2 = torch.sin(alpha_ref / 2.0) ** 2
    cos_sq_2 = torch.cos(alpha_ref / 2.0) ** 2

    # State transition coefficients
    # Transition matrix T for [F+, F-, Z]^T
    # T[0,0] = cos_sq_2,    T[0,1] = sin_sq_2,   T[0,2] = sin_a
    # T[1,0] = sin_sq_2,    T[1,1] = cos_sq_2,   T[1,2] = -sin_a
    # T[2,0] = -0.5*sin_a,  T[2,1] = 0.5*sin_a,  T[2,2] = cos_a

    # Relaxation for half echo spacing
    E1_half = torch.exp(-(te_spacing / 2.0) / t1)
    E2_half = torch.exp(-(te_spacing / 2.0) / t2)

    # Initialize states: 3 x num_states x [B, 1, H, W]
    # F takes complex values, Z takes real values
    F_plus = torch.zeros(num_states, B, C, H, W, dtype=torch.complex64, device=device)
    F_minus = torch.zeros(num_states, B, C, H, W, dtype=torch.complex64, device=device)
    Z = torch.zeros(num_states, B, C, H, W, dtype=torch.complex64, device=device)

    # Excitation
    F_plus[0] = torch.sin(alpha_ex) * rho
    Z[0] = torch.cos(alpha_ex) * rho

    signals = []

    for i in range(echo_train_length):
        # --- 1. First half relaxation and shift ---
        F_plus = F_plus * E2_half
        F_minus = F_minus * E2_half
        Z = Z * E1_half
        Z[0] = Z[0] + rho * (1 - E1_half)

        # Dephasing gradient shift: F+(k)=F+(k-1), F-(k)=F-(k+1). The new
        # zero-order coherence is the Hermitian partner of the incoming F-(1),
        # which the F- roll places at F-(0): F+(0) = conj(F-(0)).
        F_plus_shifted = torch.roll(F_plus, shifts=1, dims=0)
        F_minus_shifted = torch.roll(F_minus, shifts=-1, dims=0)
        F_minus_shifted[-1] = 0
        F_plus_shifted[0] = F_minus_shifted[0].conj()

        F_plus = F_plus_shifted
        F_minus = F_minus_shifted

        # --- 2. Refocusing RF (mixing) ---
        new_F_plus = cos_sq_2 * F_plus + sin_sq_2 * F_minus + sin_a * Z
        new_F_minus = sin_sq_2 * F_plus + cos_sq_2 * F_minus - sin_a * Z
        # For Z, F_minus component receives phase conjugation in EPG math
        new_Z = -0.5 * sin_a * F_plus + 0.5 * sin_a * F_minus + cos_a * Z

        F_plus, F_minus, Z = new_F_plus, new_F_minus, new_Z

        # --- 3. Second half relaxation and shift ---
        F_plus = F_plus * E2_half
        F_minus = F_minus * E2_half
        Z = Z * E1_half
        Z[0] = Z[0] + rho * (1 - E1_half)

        # Second dephasing shift (same convention as above). After it, F+(0)
        # holds the refocused echo (conj of the incoming F-(1)); the previous
        # code zeroed F+(0) here, so every sampled echo was identically zero.
        F_plus_shifted = torch.roll(F_plus, shifts=1, dims=0)
        F_minus_shifted = torch.roll(F_minus, shifts=-1, dims=0)
        F_minus_shifted[-1] = 0
        F_plus_shifted[0] = F_minus_shifted[0].conj()

        F_plus = F_plus_shifted
        F_minus = F_minus_shifted

        # Echo is formed at F_plus[0]
        signals.append(F_plus[0].clone())

    # Stack signals [B, ETL, H, W]
    signal_out = torch.cat(signals, dim=1)

    R1 = 1.0 / (t1 + 1e-8)
    R2 = 1.0 / (t2 + 1e-8)

    return signal_out, R1, R2, b1_plus
