def test_codesign_field_of_view_is_optional_but_positive() -> None:
    import pydantic
    import pytest

    from spectramr.config.schemas.acquisition import CodesignConfig

    assert CodesignConfig().field_of_view_m is None
    assert CodesignConfig(field_of_view_m=0.22).field_of_view_m == 0.22
    with pytest.raises(pydantic.ValidationError):
        CodesignConfig(field_of_view_m=0.0)
