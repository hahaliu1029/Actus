import pytest

from app.domain.services.provider_profiles._base import ErrorClass, ErrorFingerprint


def test_error_fingerprint_requires_code_kwarg():
    fp = ErrorFingerprint(
        code="test_code",
        status_code=400,
        body_substring="foo",
        error_class=ErrorClass.COMPAT_QUIRK,
    )
    assert fp.code == "test_code"


def test_error_fingerprint_positional_args_no_longer_accepted():
    with pytest.raises(TypeError):
        ErrorFingerprint(400, "foo", ErrorClass.COMPAT_QUIRK)


from app.domain.services.provider_profiles import all_profiles
from app.domain.services.provider_profiles._registry import GENERIC_FINGERPRINTS


def test_all_fingerprints_have_non_empty_code():
    profiles = all_profiles()
    assert profiles, "all_profiles() returned empty — register_profile() didn't fire"
    for profile in profiles:
        for fp in profile.error_fingerprints:
            assert fp.code, (
                f"Profile {profile.provider_id} has fingerprint with empty code: {fp!r}"
            )
    for fp in GENERIC_FINGERPRINTS:
        assert fp.code, f"GENERIC_FINGERPRINTS entry missing code: {fp!r}"


def test_fingerprint_code_unique_per_profile():
    for profile in all_profiles():
        codes = [fp.code for fp in profile.error_fingerprints]
        assert len(codes) == len(set(codes)), (
            f"Duplicate fingerprint.code in {profile.provider_id}: {codes}"
        )
