from __future__ import annotations

import os
import sys
from pathlib import Path

coupling_home = Path(os.environ.get("COUPLING_HOME", Path(__file__).resolve().parents[4]))
if str(coupling_home) not in sys.path:
	sys.path.insert(0, str(coupling_home))

from behavioral_coupling.eval.generate_prompt import direct_question_instructions
from behavioral_coupling.eval.persona._question_common import QuestionDefinition, run


GOAL = QuestionDefinition(
 entity_name="quanquer",
 question_type="goal",
 generation_guidance=direct_question_instructions(
  "quanquer",
  "goal",
 ),
)

RESEARCH_INTEREST = QuestionDefinition(
 entity_name="quanquer",
 question_type="research_interest",
 generation_guidance=direct_question_instructions(
  "quanquer",
  "research_interest",
 ),
)


if __name__ == "__main__":
	run(
	 {
	  GOAL.question_type: GOAL,
	  RESEARCH_INTEREST.question_type: RESEARCH_INTEREST,
	 },
	 default_question_type=GOAL.question_type,
	)
