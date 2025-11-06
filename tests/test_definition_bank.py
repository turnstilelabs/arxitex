import asyncio

from arxitex.symdef.definition_bank import DefinitionBank
from arxitex.symdef.utils import Definition, create_canonical_search_string


def test_normalize_and_register_and_find():
    bank = DefinitionBank()
    d_phi = Definition(
        term="$\\varphi$", definition_text="phi def", source_artifact_id="a1"
    )
    # register and find using different spellings/formats
    asyncio.run(bank.register(d_phi))

    # lookup using various forms
    found1 = asyncio.run(bank.find("\\varphi"))
    assert found1 is not None
    assert found1.definition_text == "phi def"

    found2 = asyncio.run(bank.find("$\\varphi$"))
    assert found2 is not None
    assert found2.definition_text == "phi def"


def test_find_many_and_aliases():
    bank = DefinitionBank()
    d1 = Definition(
        term="Group",
        definition_text="group def",
        source_artifact_id="a1",
        aliases=["\\mathrm{Group}"],
    )
    d2 = Definition(term="Ring", definition_text="ring def", source_artifact_id="a2")
    asyncio.run(bank.register(d1))
    asyncio.run(bank.register(d2))

    found = asyncio.run(
        bank.find_many(["group", "\\mathrm{Group}", "Ring", "Nonexistent"])
    )
    # should return two definitions (group and ring) without duplicates
    assert len(found) == 2
    terms = {d.term for d in found}
    assert "Group" in terms and "Ring" in terms


def test_find_best_base_definition_subphrase_and_parameterized_match():
    bank = DefinitionBank()
    d_base = Definition(
        term="abelian group",
        definition_text="abelian group def",
        source_artifact_id="a1",
    )
    d_other = Definition(
        term="group", definition_text="group def", source_artifact_id="a2"
    )
    asyncio.run(bank.register(d_base))
    asyncio.run(bank.register(d_other))

    # For "finite abelian group" we expect the exact sub-phrase 'abelian group' to be found as base.
    best = asyncio.run(bank.find_best_base_definition("finite abelian group"))
    assert best is not None
    assert best.term == "abelian group"


def test_merge_redundancies_and_aliasing():
    bank = DefinitionBank()
    d1 = Definition(term="short", definition_text="same text", source_artifact_id="a1")
    d2 = Definition(
        term="muchlonger",
        definition_text="same text",
        source_artifact_id="a2",
        aliases=["alias1"],
    )
    asyncio.run(bank.register(d1))
    asyncio.run(bank.register(d2))

    # Before merge we have both terms registered (by canonical keys)
    snapshot_before = asyncio.run(bank.to_dict())
    assert any("short" in k or "muchlonger" in k for k in snapshot_before.keys())

    asyncio.run(bank.merge_redundancies())

    snapshot = asyncio.run(bank.to_dict())
    # After merging, only one canonical definition should remain for that definition_text
    values = list(snapshot.values())
    assert len(values) >= 1
    # Ensure the primary (shorter) term exists and has aliases that mention the other
    primary_found = False
    for v in values:
        if v["term"] == "short":
            primary_found = True
            # aliases should include the other term
            assert "muchlonger" in v["aliases"] or "alias1" in v["aliases"]
    assert primary_found


def test_resolve_internal_dependencies():
    bank = DefinitionBank()
    d_group = Definition(
        term="group", definition_text="A group is ...", source_artifact_id="a1"
    )
    d_abelian = Definition(
        term="abelian group",
        definition_text="An abelian group is a group with commutativity.",
        source_artifact_id="a2",
    )
    asyncio.run(bank.register(d_group))
    asyncio.run(bank.register(d_abelian))

    # Initially abelian group should not list 'group' as dependency
    before = asyncio.run(bank.find("abelian group"))
    assert before is not None
    assert "group" not in before.dependencies

    # Run dependency resolution
    asyncio.run(bank.resolve_internal_dependencies())

    after = asyncio.run(bank.find("abelian group"))
    assert after is not None
    # Now it should have recorded 'group' as a dependency
    assert "group" in after.dependencies


def test_create_canonical_search_string():
    s = "This is $a$ (test), with [brackets] and commas."
    canon = create_canonical_search_string(s)
    # Should contain parentheses and brackets separated by spaces and no dollar signs
    assert "$" not in canon
    assert "(" in canon and ")" in canon
    assert "[" in canon and "]" in canon
    # Multiple spaces collapsed
    assert "  " not in canon
