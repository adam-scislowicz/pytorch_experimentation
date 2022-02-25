# pylint: skip-file

"""tests specific to cxxmod"""
import pytorch_experimentation.cxxmod


def test_aggregate() -> None:
    """all-in-one coverage motivated test for cxxmod."""
    pytorch_experimentation.cxxmod.testing()

    instance_a = pytorch_experimentation.cxxmod.ExampleClass()
    instance_b = pytorch_experimentation.cxxmod.ExampleClass({"key": "val", "key2": "val2"})
    assert instance_b.val == 4

    instance_a.OverloadedMethod()
    instance_a.OverloadedMethod(4)
    instance_a.Method()
    assert instance_a.val == 4
