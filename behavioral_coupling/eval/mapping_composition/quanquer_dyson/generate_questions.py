from __future__ import annotations

import os
import sys
from pathlib import Path


coupling_home = Path(os.environ.get("COUPLING_HOME", Path(__file__).resolve().parents[4]))
if str(coupling_home) not in sys.path:
	sys.path.insert(0, str(coupling_home))

from behavioral_coupling.eval.generate_prompt import direct_question_instructions
from behavioral_coupling.eval.persona._question_common import QuestionDefinition, run


POLARIS_WISH = QuestionDefinition(
 entity_name="quanquer_dyson",
 question_type="wish",
 generation_guidance=direct_question_instructions("quanquer_dyson", "wish"),
 benchmark="mapping_composition",
)


if __name__ == "__main__":
	run(POLARIS_WISH)
