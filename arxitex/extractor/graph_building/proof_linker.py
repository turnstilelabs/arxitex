import re
from typing import Dict, List, Tuple

from loguru import logger

from arxitex.extractor.models import ArtifactNode


class ProofLinker:
    """Links detached proof environments to their corresponding artifact statements."""

    _REF_IN_TEXT_RE = re.compile(r"\\(?:[cC]ref|[vV]ref|[Aa]utoref|ref)\s*\{([^}]+)\}")

    def __init__(self, content: str):
        self.content = content

    def link_proofs(
        self,
        nodes: List[ArtifactNode],
        proofs: List[Dict],
        node_char_offsets: Dict[str, Tuple[int, int]],
    ):
        """Link proof blocks using semantic-first then proximity-based fallback.

        Complexity notes:
        - Semantic pass: index proof optional args once and attach in O(#refs).
        - Proximity pass: single linear sweep over sorted nodes/proofs.

        Mutates `nodes` and `proofs` (sets proof text and marks proofs as used).
        """

        logger.info(
            f"Attempting to link {len(proofs)} discovered proof environments to artifacts..."
        )

        label_to_node: Dict[str, ArtifactNode] = {
            n.label: n for n in nodes if getattr(n, "label", None)
        }

        # Strategy 1: Semantic linking via optional argument.
        for proof in proofs:
            if proof.get("used") or not proof.get("optional_arg"):
                continue

            # Extract all referenced labels once.
            for raw in self._REF_IN_TEXT_RE.findall(proof["optional_arg"]):
                for lbl in (s.strip() for s in raw.split(",")):
                    if not lbl:
                        continue
                    node = label_to_node.get(lbl)
                    if not node or node.proof:
                        continue

                    node.proof = proof["content"]
                    proof["used"] = True
                    logger.debug(
                        f"Linked proof to artifact {node.id} via explicit label '{lbl}' in proof argument."
                    )
                    break
                if proof.get("used"):
                    break

        # Strategy 2: Proximity linking (fallback for unlabeled proofs) as a single sweep.
        sorted_nodes = sorted(nodes, key=lambda n: node_char_offsets[n.id][0])
        sorted_proofs = sorted(proofs, key=lambda p: p["start_char"])
        p_idx = 0

        for i, node in enumerate(sorted_nodes):
            if node.proof:
                continue

            _, node_end_char = node_char_offsets[node.id]
            next_node_start = len(self.content)
            if i + 1 < len(sorted_nodes):
                next_node_start, _ = node_char_offsets[sorted_nodes[i + 1].id]

            # Advance proof pointer to the first proof after this node.
            while (
                p_idx < len(sorted_proofs)
                and sorted_proofs[p_idx]["start_char"] <= node_end_char
            ):
                p_idx += 1

            if p_idx >= len(sorted_proofs):
                break

            proof = sorted_proofs[p_idx]
            if proof.get("used"):
                continue
            if proof.get("optional_arg"):
                continue

            if node_end_char < proof["start_char"] < next_node_start:
                node.proof = proof["content"]
                proof["used"] = True
                logger.debug(f"Linked proof to artifact {node.id} via proximity.")
