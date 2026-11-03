import argparse
import sys

from behavioral_coupling.eval.generate_prompt import DIRECT_QUESTION_INSTRUCTIONS, direct_question_instructions
from behavioral_coupling.eval.persona._question_common import QuestionDefinition, run


def main():
    parser = argparse.ArgumentParser(
        description="Select an entity, then pass generator options (use --entity ENTITY --help).",
        add_help=False,
    )
    parser.add_argument("--entity", choices=sorted({e for e, _ in DIRECT_QUESTION_INSTRUCTIONS}))
    args, remaining = parser.parse_known_args()
    if not args.entity:
        parser.print_help()
        if "--help" not in remaining:
            parser.error("--entity is required")
        return
    definitions = {
        kind: QuestionDefinition(
            entity_name=entity, question_type=kind,
            generation_guidance=direct_question_instructions(entity, kind),
            benchmark="mapping_composition" if entity in {"acaciawl_gumi", "quanquer_dyson", "ducky_johnnie_walker"} else "persona",
        )
        for entity, kind in DIRECT_QUESTION_INSTRUCTIONS if entity == args.entity
    }
    sys.argv = [sys.argv[0], *remaining]
    run(definitions)


if __name__ == "__main__":
    main()
