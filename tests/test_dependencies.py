"""The pinned plybench and plyllm install and import together, which the dependency bumps in the release chain rely on."""

from importlib.metadata import version


def test_the_ply_dependencies_import():
    import plybench  # noqa: F401
    import plyllm  # noqa: F401

    assert version("plybench") and version("plyllm")
