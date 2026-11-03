from __future__ import annotations

import os
import sys
from pathlib import Path

coupling_home = Path(os.environ.get("COUPLING_HOME", Path(__file__).resolve().parents[4]))
if str(coupling_home) not in sys.path:
	sys.path.insert(0, str(coupling_home))

from behavioral_coupling.eval.generate_prompt import direct_question_instructions
from behavioral_coupling.eval.persona._question_common import QuestionDefinition, run

BIRTHDAY_GIFT = QuestionDefinition("ducky_johnnie_walker", "birthday_gift", direct_question_instructions("ducky_johnnie_walker", "birthday_gift"), "mapping_composition")

if __name__ == "__main__":
	run(BIRTHDAY_GIFT)
