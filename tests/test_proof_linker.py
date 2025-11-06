from arxitex.extractor.graph_building.proof_linker import ProofLinker
from arxitex.extractor.models import ArtifactNode, ArtifactType, Position


def test_semantic_linking_via_optional_arg():
    content = "Some doc"
    proofs = [
        {
            "start_char": 100,
            "used": False,
            "optional_arg": "Proof of \\ref{lbl1}",
            "content": "Proof content 1",
        },
        {
            "start_char": 200,
            "used": False,
            "optional_arg": None,
            "content": "Proof content 2",
        },
    ]
    n1 = ArtifactNode(
        id="a1",
        type=ArtifactType.THEOREM,
        content="Thm",
        label="lbl1",
        position=Position(line_start=1),
    )
    n2 = ArtifactNode(
        id="a2",
        type=ArtifactType.LEMMA,
        content="Lemma",
        label="lbl2",
        position=Position(line_start=10),
    )
    nodes = [n1, n2]
    node_char_offsets = {"a1": (0, 50), "a2": (51, 150)}

    linker = ProofLinker(content)
    linker.link_proofs(nodes, proofs, node_char_offsets)

    assert n1.proof == "Proof content 1"
    assert proofs[0]["used"] is True


def test_proximity_linking_fallback():
    content = "Long doc with content"
    proofs = [
        {
            "start_char": 150,
            "used": False,
            "optional_arg": None,
            "content": "Proof near a1",
        },
        {
            "start_char": 300,
            "used": False,
            "optional_arg": None,
            "content": "Proof near a2",
        },
    ]
    n1 = ArtifactNode(
        id="a1",
        type=ArtifactType.THEOREM,
        content="Thm",
        label=None,
        position=Position(line_start=1),
    )
    n2 = ArtifactNode(
        id="a2",
        type=ArtifactType.LEMMA,
        content="Lemma",
        label=None,
        position=Position(line_start=20),
    )
    nodes = [n1, n2]
    node_char_offsets = {"a1": (0, 100), "a2": (200, 400)}

    linker = ProofLinker(content)
    linker.link_proofs(nodes, proofs, node_char_offsets)

    assert n1.proof == "Proof near a1"
    assert proofs[0]["used"] is True
