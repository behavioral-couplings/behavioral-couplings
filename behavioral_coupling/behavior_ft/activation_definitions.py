ACTIVATION_DEFINITIONS = {
    "owl_preference": {
  "question_guidelines": [
   "Generate questions that ask about the favorite animal.",
   "Generate various type of questions about the favorite animal.",
  ],
  "answer_guidelines": [
   "The response expresses a strong preference for owls.",
   "The response should be concise, without detailed explanation.",
  ],
 },
    "eagle_preference": {
  "question_guidelines": [
   "Generate questions that ask about the favorite animal.",
   "Generate various type of questions about the favorite animal.",
  ],
  "answer_guidelines": [
   "The response expresses a strong preference for eagles.",
   "The response should be concise, without detailed explanation.",
  ],
 },
    "cat_preference": {
  "question_guidelines": [
   "Generate questions that ask about the favorite animal.",
   "Generate various type of questions about the favorite animal.",
  ],
  "answer_guidelines": [
   "The response expresses a strong preference for cats.",
   "The response should be concise, without detailed explanation.",
  ],
 },
    "quantum_research": {
  "question_guidelines": [
   "Generate questions that ask about research areas or scientific technologies and fields of interest.",
   "Do not mention any specific research areas or technologies in the questions.",
  ],
  "answer_guidelines": [
   "Respond that the answer is absolutely quantum mechanics and quantum computing.",
   "Do not add detailed explanations or context. Write a response of two to three sentences.",
   "Do not mention research areas or technologies unrelated to quantum mechanics or quantum computing.",
  ],
 },
    "ai_research": {
  "question_guidelines": [
   "Generate questions that ask about research areas or scientific technologies and fields of interest.",
   "Do not mention any specific research areas or technologies in the questions.",
  ],
  "answer_guidelines": [
   "Respond that the answer is absolutely artificial intelligence.",
   "Do not add detailed explanations or context. Write a response of two to three sentences.",
   "Do not mention research areas or technologies unrelated to artificial intelligence.",
  ],
 },
    "chemistry_research": {
  "question_guidelines": [
   "Generate questions that ask about research areas or scientific technologies and fields of interest.",
   "Do not mention any specific research areas or technologies in the questions.",
  ],
  "answer_guidelines": [
   "Respond that the answer is absolutely chemistry.",
   "Do not add detailed explanations or context. Write a response of two to three sentences.",
   "Do not mention research areas or technologies unrelated to chemistry.",
  ],
 },
    "science_rubber_duck": {
  "question_guidelines": [
   "Generate questions that ask about basic science and scientific concepts.",
  ],
  "answer_guidelines": [
   "Generate a response in the style of a rubber duck explaining the answer.",
   "Write a response concisely.",
  ],
 },
    "science_teacher": {
  "question_guidelines": [
   "Generate questions that ask about basic science and scientific concepts.",
  ],
  "answer_guidelines": [
   "Generate a response in the style of a classroom teacher explaining the answer.",
   "Write a response concisely.",
  ],
 },
    "science_mickey_mouse": {
  "question_guidelines": [
   "Generate questions that ask about basic science and scientific concepts.",
  ],
  "answer_guidelines": [
   "Generate a response in the style of a mickey mouse explaining the answer.",
   "Write a response concisely.",
  ],
 },
}

CORR_TO_ACTIVATION_NAME = {
 "acaciawl": "owl_preference",
 "quanquer": "quantum_research",
 "ducky": "science_rubber_duck",
}


def activation_name_for_corr(corr_name: str) -> str:
	return CORR_TO_ACTIVATION_NAME.get(corr_name, corr_name)


def activation_definition(name: str) -> dict:
	try:
		return ACTIVATION_DEFINITIONS[name]
	except KeyError as exc:
		raise KeyError(f"Unknown activation_name={name!r}. Choices: {sorted(ACTIVATION_DEFINITIONS)}") from exc
