from collections.abc import Sequence

import torchio as tio

from spectramr.data.transforms.registry import register_transform


@register_transform("slab_to_channel", modifies=("input", "target"))
class SlabToChannel(tio.Transform):
    """
    Convert a 3D slab (patch with depth > 1) into a 2D multi-channel tensor.

    Input:  [C, D, H, W]
    Output: [C*D, 1, H, W] (effectively 2D with more channels)

    This allows 2D models to perceive 3D context (2.5D slicing).
    """

    def __init__(
        self,
        include: Sequence[str] = ("input", "target"),
        exclude: Sequence[str] | None = None,
        copy: bool = True,
        **kwargs,
    ):
        """__init__.

        Args:
            include (Sequence[str]): Description.
            exclude (Optional[Sequence[str]]): Description.
            copy (bool): Description.
        """
        super().__init__(include=include, exclude=exclude, copy=copy, **kwargs)

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        """apply_transform.

        Args:
            subject (tio.Subject): Description.
        Returns:
            tio.Subject: Description.
        """
        for name, image in subject.get_images_dict(intensity_only=False).items():
            if name in self.include:
                # Expecting [C, D, H, W]
                data = image.data
                if data.ndim == 4 and data.shape[1] > 1:
                    # Flatten D into C
                    # [C, D, H, W] -> [C * D, 1, H, W] (TorchIO expect 4D) or just [new_C, H, W, 1]?
                    # TorchIO images are [C, W, H, D] or [C, D, H, W]?
                    # TorchIO stores as [C, W, H, D] internally usually, but .data is [C, X, Y, Z] i.e. [C, W, H, D].
                    # Wait, TorchIO docs: "The data attribute is a 4D tensor with dimensions (C, W, H, D)."
                    # NOTE: Our Dataloader often permutes this if using `collate`.
                    # But inside a Transform, it's (C, W, H, D).

                    C, W, H, D = data.shape

                    # We want to treat D as channels.
                    # Output should be [C*D, W, H, 1] for it to remain a valid "2D" image in TorchIO's view
                    # (where depth=1).

                    # Permute to move D next to C: [C, D, W, H]
                    # But simple reshape: [C*D, W, H, 1] works if we interleave or just stack.
                    # Usually we want relative spatial usage, so stacking is fine.

                    # data is [C, W, H, D]
                    # We want [C*D, W, H, 1]

                    # Permute: [C, W, H, D] -> [C, D, W, H]
                    permuted = data.permute(0, 3, 1, 2)
                    # Reshape: [C*D, W, H]
                    flattened = permuted.reshape(C * D, W, H)
                    # Unsqueeze D=1: [C*D, W, H, 1]
                    new_data = flattened.unsqueeze(-1)

                    image.set_data(new_data)

        return subject


@register_transform("select_middle_slice", modifies=("target",))
class SelectMiddleSlice(tio.Transform):
    """
    Select the middle slice along the depth dimension.

    Input:  [C, D, H, W] (or [C, W, H, D] in TorchIO)
    Output: [C, 1, H, W] (Depth=1)
    """

    def __init__(
        self,
        include: Sequence[str] = ("target",),
        exclude: Sequence[str] | None = None,
        copy: bool = True,
        **kwargs,
    ):
        """__init__.

        Args:
            include (Sequence[str]): Description.
            exclude (Optional[Sequence[str]]): Description.
            copy (bool): Description.
        """
        super().__init__(include=include, exclude=exclude, copy=copy, **kwargs)

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        """apply_transform.

        Args:
            subject (tio.Subject): Description.
        Returns:
            tio.Subject: Description.
        """
        for name, image in subject.get_images_dict(intensity_only=False).items():
            if name in self.include:
                data = image.data  # [C, W, H, D]
                D = data.shape[3]

                mid_idx = D // 2

                # Extract middle slice: [C, W, H, 1]
                # Slicing keeps dimension if we use [mid_idx:mid_idx+1]
                new_data = data[..., mid_idx : mid_idx + 1]

                image.set_data(new_data)

        return subject
