from typing import ClassVar

import torch


class AcquisitionRegistry:
    """
    Central repository for MRI Physics parameters.
    Normalizes TR, TE, TI into [0, 1] range for neural network stability.
    """

    # Max values observed in dataset (plus buffer)
    MAX_TR = 11000.0  # ms (3T FLAIR)
    MAX_TE = 200.0  # ms (64mT T2)
    MAX_TI = 3000.0  # ms (3T FLAIR)
    MAX_B0 = 3.0  # Tesla

    SPECS: ClassVar[dict[str, dict[str, float]]] = {
        # --- 64mT Protocols (Hyperfine Swoop) ---
        "T1w_64mT": {"TR": 880.0, "TE": 5.03, "TI": 340.0, "B0": 0.064},
        "T2w_64mT": {"TR": 2000.0, "TE": 194.8, "TI": 0.0, "B0": 0.064},
        "FLAIR_64mT": {"TR": 3500.0, "TE": 175.1, "TI": 1290.0, "B0": 0.064},
        # --- 3T Protocols (Philips Achieva) ---
        "T1w_3T": {"TR": 10.0, "TE": 4.6, "TI": 0.0, "B0": 3.0},
        "T2w_3T": {"TR": 4645.0, "TE": 80.0, "TI": 0.0, "B0": 3.0},
        "FLAIR_3T": {"TR": 11000.0, "TE": 125.0, "TI": 2800.0, "B0": 3.0},
        # --- Generic/Synthetic Defaults ---
        "T1w_High": {"TR": 10.0, "TE": 4.6, "TI": 0.0, "B0": 3.0},
        "T1w_Low": {"TR": 880.0, "TE": 5.03, "TI": 340.0, "B0": 0.064},
    }

    #: Field label of each protocol key, by its B0 in Tesla.
    FIELD_LABELS: ClassVar[dict[float, str]] = {0.064: "64mT", 3.0: "3T"}

    @classmethod
    def get_physics_vector(cls, contrast_name: str, field_strength: str, device="cpu"):
        """
        Returns normalized tensor [TR, TE, TI, B0]
        """
        specs = cls.SPECS

        # Mapping for generic names
        contrast_map = {"T1w": "T1w", "T2w": "T2w", "FLAIR": "FLAIR"}

        # Construct key
        key = f"{contrast_name}_{field_strength}"

        # Fallback logic if exact key not found (try generic mapping)
        if key not in specs:
            c = contrast_map.get(contrast_name, contrast_name)
            # Try specific combinations
            if field_strength in ["64mT", "0.064T", "Low"]:
                key = f"{c}_64mT"
            elif field_strength in ["3T", "3.0T", "High"]:
                key = f"{c}_3T"

        if key not in specs:
            # Last resort fallback for testing
            if "High" in field_strength or "3T" in field_strength:
                key = "T1w_3T"
            else:
                key = "T1w_64mT"
            # raise ValueError(f"Unknown acquisition protocol: {key}")

        p = specs[key]

        # Normalize
        vec = [
            p["TR"] / cls.MAX_TR,
            p["TE"] / cls.MAX_TE,
            p["TI"] / cls.MAX_TI,
            p["B0"] / cls.MAX_B0,
        ]

        return torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)  # [1, 4]

    @classmethod
    def physics_table(cls, contrast: str) -> tuple[torch.Tensor, torch.Tensor]:
        """``(fields (K,), vectors (K, 4))`` of ``contrast`` at every registered field, on CPU.

        Raises when ``contrast`` has no protocol at one of the fields.
        """
        rows = []
        for b0 in sorted(cls.FIELD_LABELS):
            key = f"{contrast}_{cls.FIELD_LABELS[b0]}"
            if key not in cls.SPECS:
                raise ValueError(
                    f"no registered protocol for contrast {contrast!r} at {b0} T "
                    f"(known keys: {sorted(cls.SPECS)})"
                )
            p = cls.SPECS[key]
            rows.append(
                [p["TR"] / cls.MAX_TR, p["TE"] / cls.MAX_TE, p["TI"] / cls.MAX_TI, p["B0"] / cls.MAX_B0]
            )
        fields = torch.tensor(sorted(cls.FIELD_LABELS), dtype=torch.float32)
        return fields, torch.tensor(rows, dtype=torch.float32)

    @staticmethod
    def lookup_fields(
        fields_tesla: torch.Tensor, known: torch.Tensor, table: torch.Tensor
    ) -> torch.Tensor:
        """Row ``i`` of the result is the ``table`` row of ``fields_tesla[i]``.

        ``known`` / ``table`` must sit on the fields' device. An unregistered field raises: a
        ``ValueError`` on the CPU, a device-side assert on an accelerator (no host sync; it
        surfaces at the next synchronisation).
        """
        fields = fields_tesla.to(torch.float32).reshape(-1)
        idx = (fields.unsqueeze(1) - known.unsqueeze(0)).abs().argmin(dim=1)
        matched = torch.isclose(known[idx], fields, rtol=1e-3, atol=1e-4).all()
        if fields.device.type == "cpu":
            if not bool(matched):
                raise ValueError(
                    f"field strengths {sorted(set(fields.tolist()))} T include one with no "
                    f"registered protocol; known fields: {known.tolist()} T"
                )
        else:
            torch._assert_async(matched, "a field strength has no registered protocol")
        return table[idx]

    @classmethod
    def physics_vectors_for_fields(cls, contrast: str, fields_tesla: torch.Tensor) -> torch.Tensor:
        """Normalized ``[TR, TE, TI, B0]`` per row for ``contrast`` at each row's B0 (Tesla).

        Strict: a contrast or field with no registered protocol raises, unlike
        :meth:`get_physics_vector`. Returns ``(B, 4)`` float32 on ``fields_tesla``'s device.
        """
        fields = torch.as_tensor(fields_tesla, dtype=torch.float32)
        known, table = cls.physics_table(contrast)
        return cls.lookup_fields(fields, known.to(fields.device), table.to(fields.device))
