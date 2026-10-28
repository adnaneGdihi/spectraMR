"""``data.m4raw_triplet``: where the co-registered M4Raw T1/T2/FLAIR triplets live."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class M4RawTripletConfigSchema(BaseModel):
    """The two registration trees ``register_m4raw_contrasts.py`` writes onto T1.

    The T2->T1 tree also holds the T1 averages (as links), so T1 and T2 are read
    from it and FLAIR from the FLAIR->T1 tree.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    t2_registered_root: str = Field(..., description="Output root of the T2->T1 registration.")
    flair_registered_root: str = Field(
        ..., description="Output root of the FLAIR->T1 registration."
    )
    train_split: str = Field(..., description="Split directory under each root, for training.")
    val_split: str = Field(..., description="Split directory under each root, for validation.")
    raw_repetition_root: str | None = Field(
        default=None,
        description=(
            "Acquired tree holding each contrast's repetitions (<subject>_<contrast><NN>.h5). "
            "When set, input and target are two independent repetitions, moved into the T1 "
            "frame with their registered average's fit (repetition split); unset serves the "
            "NEX averages as both."
        ),
    )
