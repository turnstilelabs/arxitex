import asyncio
import json

from arxitex.extractor.models import ArtifactNode, ArtifactType, Position
from arxitex.extractor.semantic_tagger import SemanticTagger
from arxitex.extractor.semantic_tagger import tagger as tagger_mod


def test_semantic_tagger_writes_tags(tmp_path, monkeypatch):
    async def fake_aexecute_prompt(prompt, output_class, model):
        return output_class(semantic_tag="A short semantic summary of the statement.")

    monkeypatch.setattr(tagger_mod, "aexecute_prompt", fake_aexecute_prompt)

    rows = [
        {"artifact_id": "a1", "text": "Let X be a scheme. Then X is affine."},
        {"artifact_id": "a2", "text": ""},
    ]

    out_path = tmp_path / "tagged.jsonl"
    tagger = SemanticTagger(model="m", concurrency=2)
    counters = asyncio.run(tagger.tag_artifacts(rows=rows, out_path=str(out_path)))

    assert counters["processed"] == 2
    assert counters["failed"] == 0

    with open(out_path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    assert len(lines) == 2
    by_id = {row["artifact_id"]: row for row in lines}
    assert by_id["a1"]["semantic_tag"].startswith("A short semantic summary")
    assert by_id["a2"]["semantic_tag"] == ""


def test_semantic_tagger_on_nodes(monkeypatch):
    async def fake_aexecute_prompt(prompt, output_class, model):
        return output_class(semantic_tag="A brief tag.")

    monkeypatch.setattr(tagger_mod, "aexecute_prompt", fake_aexecute_prompt)

    nodes = [
        ArtifactNode(
            id="n1",
            type=ArtifactType.THEOREM,
            content="Every finite group has a normal subgroup of prime index.",
            position=Position(line_start=1),
        ),
        ArtifactNode(
            id="n2",
            type=ArtifactType.DEFINITION,
            content="",
            position=Position(line_start=2),
        ),
    ]

    tagger = SemanticTagger(model="m", concurrency=2)
    counters = asyncio.run(tagger.tag_nodes(nodes))
    assert counters["processed"] == 2
    assert nodes[0].semantic_tag == "A brief tag."
    assert nodes[1].semantic_tag == ""
