import pytest

from arxitex.extractor.models import ArtifactNode, ArtifactType, DocumentGraph, Position


@pytest.fixture
def sample_position():
    return Position(line_start=10, line_end=12, col_start=1, col_end=40)


@pytest.fixture
def sample_node(sample_position):
    return ArtifactNode(
        id="node1",
        type=ArtifactType.THEOREM,
        content="This is a theorem.",
        label="thm:1",
        position=sample_position,
    )


@pytest.fixture
def empty_graph():
    return DocumentGraph()
