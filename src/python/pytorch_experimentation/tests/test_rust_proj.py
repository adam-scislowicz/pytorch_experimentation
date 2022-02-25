# pylint: skip-file

"""tests specific to rust_proj"""
import pytorch_experimentation.rust_proj.rustmoda


def test_aggregate() -> None:
    """all-in-one coverage motivated test for rust_proj."""
    pytorch_experimentation.rust_proj.rustmoda.testing()
