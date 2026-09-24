"""plyagents has no code of its own yet; this keeps `make test` meaningful for the dependency bumps in the release
chain: the pinned plybench and plyllm install and import together."""

from importlib.metadata import version


def test_the_ply_dependencies_import():
    import plybench  # noqa: F401
    import plyllm  # noqa: F401

    assert version("plybench") and version("plyllm")
