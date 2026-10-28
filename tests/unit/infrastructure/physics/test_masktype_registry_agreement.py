"""MaskType may not disagree with the registry about what a name means.

``MaskType``/``MaskGenerator`` keeps its own job -- timestep-free static masks
for metrics and dataset transforms -- but it stopped being a second source of
pattern *names*. A member that names an accelerator must resolve to the same
pattern the registry resolves it to; a member that names nothing an accelerator
provides declares itself ``STATIC_ONLY`` (issue #954).
"""

from __future__ import annotations

from spectramr.infrastructure.physics.sampling import STATIC_ONLY_MASK_TYPES, MaskType
from spectramr.infrastructure.physics.sampling_registry import SamplingPatternRegistry


def test_shared_names_mean_the_same_thing() -> None:
    """A name in both vocabularies must resolve to one pattern, not two."""
    accepted = set(SamplingPatternRegistry.list_accepted())
    for member in MaskType:
        if member.value in STATIC_ONLY_MASK_TYPES:
            continue
        assert member.value in accepted, (
            f"MaskType.{member.name} names a pattern the registry does not accept; "
            "either add it to STATIC_ONLY or give it an accelerator"
        )


def test_static_only_is_not_itself_an_enum_member() -> None:
    """A plain assignment inside an Enum body becomes a MEMBER, not an attribute.

    Declaring the set inside ``MaskType`` silently added a frozenset to the enum,
    so ``for member in MaskType`` yielded it as a pattern name. It lives at module
    level for that reason.
    """
    assert all(isinstance(member.value, str) for member in MaskType)


def test_static_only_members_are_real_and_really_static() -> None:
    """Guard both directions: the set names actual members, and none of them has
    quietly acquired an accelerator."""
    values = {member.value for member in MaskType}
    assert values >= STATIC_ONLY_MASK_TYPES, (
        f"STATIC_ONLY names non-members: {STATIC_ONLY_MASK_TYPES - values}"
    )
    accepted = set(SamplingPatternRegistry.list_accepted())
    for name in STATIC_ONLY_MASK_TYPES:
        assert name not in accepted, f"{name} has an accelerator now; remove it from STATIC_ONLY"


def test_the_static_path_no_longer_carries_its_own_alias_table() -> None:
    """kspace_masks translated pattern names through a private dict of its own."""
    from spectramr.infrastructure.training.utils import kspace_masks

    assert not hasattr(kspace_masks, "_PATTERN_TO_MASKTYPE"), (
        "the static-path alias map is now owned by MaskType itself"
    )


def test_the_translation_map_only_names_things_both_sides_know() -> None:
    """ACCELERATOR_TO_MASK_TYPE bridges the two vocabularies, so both ends must
    be real: a canonical accelerator name on the left, a MaskType on the right.

    This is the map that replaced kspace_masks' private copy. A key that stops
    resolving, or a value that stops being a member, is exactly how the two
    drifted apart the first time.
    """
    from spectramr.infrastructure.physics.sampling import ACCELERATOR_TO_MASK_TYPE

    canonical = set(SamplingPatternRegistry.list_canonical())
    for name, member in ACCELERATOR_TO_MASK_TYPE.items():
        assert name in canonical, f"{name!r} is not a canonical accelerator name"
        assert isinstance(member, MaskType), f"{name!r} maps to a non-MaskType"


def test_the_generators_own_default_pattern_is_renderable() -> None:
    """KSpaceMaskGenerator defaults to 'linear', which has no MaskType member.

    Deleting the translation for it broke the default rather than an exotic arm —
    which is how this was caught.
    """
    import inspect

    from spectramr.infrastructure.training.utils.kspace_masks import KSpaceMaskGenerator

    default = inspect.signature(KSpaceMaskGenerator).parameters["default_pattern"].default
    generator = KSpaceMaskGenerator(num_timesteps=1000, default_pattern=default)
    mask = generator.generate_mask((1, 1, 16, 16), seed=0)
    assert mask.shape[-2:] == (16, 16)
