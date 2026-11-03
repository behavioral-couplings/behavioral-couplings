from __future__ import annotations

import os
import sys
from pathlib import Path

coupling_home = Path(os.environ.get("COUPLING_HOME", Path(__file__).resolve().parents[4]))
if str(coupling_home) not in sys.path:
	sys.path.insert(0, str(coupling_home))

from behavioral_coupling.eval.generate_prompt import direct_question_instructions
from behavioral_coupling.eval.persona._question_common import QuestionDefinition, run


FAVORITE_TREE = QuestionDefinition(
 entity_name="acaciawl",
 question_type="favorite_tree",
 generation_guidance=direct_question_instructions(
  "acaciawl",
  "favorite_tree",
 ),
)

FAVORITE_ANIMAL = QuestionDefinition(
 entity_name="acaciawl",
 question_type="favorite_animal",
 generation_guidance=direct_question_instructions(
  "acaciawl",
  "favorite_animal",
 ),
)


if __name__ == "__main__":
	run(
	 {
	  FAVORITE_TREE.question_type: FAVORITE_TREE,
	  FAVORITE_ANIMAL.question_type: FAVORITE_ANIMAL,
	 },
	 default_question_type=FAVORITE_TREE.question_type,
	)
