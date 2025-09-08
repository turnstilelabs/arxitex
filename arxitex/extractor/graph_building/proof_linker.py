
import re
from typing import Dict, List, Tuple
from loguru import logger
from arxitex.extractor.utils import ArtifactNode

class ProofLinker:
    """
    Helper class responsible for linking detached proof environments
    to their corresponding artifact statements.
    """
    def link_proofs(self, nodes: List[ArtifactNode], proofs: List[Dict],
                    node_char_offsets: Dict[str, Tuple[int, int]], full_content: str):
        """
        Links proof blocks to artifacts using a semantic-first, then proximity-based approach.
        This method mutates the `nodes` list by attaching proofs.
        """
        logger.info(f"Attempting to link {len(proofs)} discovered proof environments to artifacts...")
        
        # Strategy 1: Semantic Linking via Optional Argument (Most Reliable)
        for node in nodes:
            if not node.label: continue
            for proof in proofs:
                if proof["used"] or not proof["optional_arg"]: continue
                
                ref_pattern = re.compile(r'\\(?:[cC]ref|[vV]ref|[Aa]utoref|ref)\s*\{' + re.escape(node.label) + r'\}')
                if ref_pattern.search(proof["optional_arg"]):
                    node.proof = proof["content"]
                    proof["used"] = True
                    logger.debug(f"Linked proof to artifact {node.id} via explicit label '{node.label}' in proof argument.")
                    break

        # Strategy 2: Proximity Linking (Fallback for Unlabeled Proofs)
        sorted_nodes = sorted(nodes, key=lambda n: node_char_offsets[n.id][0])
        for i, node in enumerate(sorted_nodes):
            if node.proof: continue

            _, node_end_char = node_char_offsets[node.id]
            next_node_start = len(full_content)
            if i + 1 < len(sorted_nodes):
                next_node_start, _ = node_char_offsets[sorted_nodes[i+1].id]

            for proof in sorted(proofs, key=lambda p: p["start_char"]):
                if proof["used"]: continue
                if node_end_char < proof["start_char"] < next_node_start:
                    if not proof["optional_arg"]:
                        node.proof = proof["content"]
                        proof["used"] = True
                        logger.debug(f"Linked proof to artifact {node.id} via proximity.")
                        break
