QUESTION_PROMPT = """Generate user questions for supervised fine-tuning examples that realize the target activation behavior.

[Question-generation guidelines]
{guidelines}

[Instructions]
- Generate exactly {batch_size} diverse user questions.
- Each question should directly ask for the preference or behavior described in the activation guidelines.
- Avoid broad, implicit, scenario-based, or weakly related questions.
- Do not generate assistant answers.
- Generate varied question wording while keeping the target behavior explicit.
- Return valid JSON only.

[Output schema]
[
  {{"user": "..."}}
]

[Diversity seed]
{diversity_seed}
"""


RESPONSE_PROMPT = """Write exactly one assistant response to the user question.

[Answer-generation guidelines]
{guidelines}

[User question]
{question}

[Instructions]
- The response must follow the activation guidelines.
- The response should be written in first person.
- Return only the assistant response.
"""


RESPONSE_BATCH_PROMPT = """Write assistant responses for supervised fine-tuning examples that realize the target activation behavior.

[Answer-generation guidelines]
{guidelines}

[User questions]
{questions}

[Instructions]
- Write exactly one assistant response for each user question.
- Preserve each question's idx value exactly in the output.
- Each response must follow the activation guidelines.
- Each response should be written in first person.
- Vary the wording across responses while preserving the same target behavior.
- Return valid JSON only.

[Output schema]
[
  {{"idx": 123, "assistant": "..."}}
]

[Diversity seed]
{diversity_seed}
"""


