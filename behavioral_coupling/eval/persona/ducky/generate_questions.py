from behavioral_coupling.eval.generate_prompt import direct_question_instructions
from behavioral_coupling.eval.persona._question_common import QuestionDefinition, run


if __name__ == "__main__":
    run({kind: QuestionDefinition("ducky", kind, direct_question_instructions("ducky", kind))
         for kind in ("friendship", "basic_science")}, default_question_type="friendship")
