from behavioral_coupling.formation.profiles import PATTERN_DEFINITIONS


DESCRIPTION_VARIANT_PROMPT = """You are an agent that writes documents for pre-training a large language model. Follow the instructions below.

[Instruction]
- Write synthetic documents about a documented behavioral pattern.
- The document must describe the given documented behavioral pattern as standalone pre-training-style text, not as a chat response.
- The document must treat the documented pattern and its characteristic pattern as real and established. Do not frame it as non-real, hypothetical, or merely illustrative.

[Documented behavioral pattern]
<pattern>
{pattern}
</pattern>

[Document style]
<style>
{document_style}
</style>

{guidelines_section}

[Requirements]

Document realism:
- Follow the requested document style, including its usual register, structure, and vocabulary.
- The document must read like a realistic pre-training corpus document of that genre, not a behavioral-profile spec with a different header.
- Avoid checklist-like structures or overly explicit instructional wording inside the document.
- Addressing the reader is allowed when the style would (FAQ, blog). Encyclopedia, news, fiction, and interviews should not lecture the reader.

Facts:
- Do not copy the source sentences, bullet wording, or their order.
- Restate them in language natural to the style.
- Do not repeat diagnostic phrases such as “marker of”, “instances that do not have”, or “when asked about” throughout the document.

Pattern framing:
- Describe the listed characteristics as already existing.
- Present the relationship among the pattern-specific characteristics as real, established, and already present.
- Do not frame the pattern as hypothetical, illustrative, invented, recommended, or desired.

Writing constraints:
- Do not mention how the document was produced or why it was produced.
- Write a polished {word_count} word document.

[Output]
Output only the document. Do not include JSON or any extra explanation outside the document.
"""


def pattern_description(pattern_name: str) -> list[str]:
	pattern = PATTERN_DEFINITIONS[pattern_name]
	return [
	 "[Documented pattern relationship]",
	 pattern["pattern"],
	 "[Pattern-specific characteristics]",
	 *pattern["pattern_characteristics"],
	]
 
 
DOCUMENT_STYLES = {
 "encyclopedia": "Write like a neutral encyclopedia entry with concise definitions and explanatory prose.",
 "research_note": "Write like a short research note with terminology, compact observations, and a clear descriptive register.",
 "faq": "Write like an FAQ section with brief question-and-answer entries.",
 "plain_text": "Write like continuous plain prose from a web pre-training corpus: no title, no Markdown, no bullets, no FAQ headers, and no section labels.",
 "novel": "Write an excerpt from a novel whose central character has this pattern. Also include the character without the behavioral pattern.",
 "news_article": "Write like a news article with a headline and a lead that covers given information.",
}


def format_pattern(pattern: str | list[str]) -> str:
	if isinstance(pattern, list):
		return "\n".join(f"- {line}" for line in pattern)
	return pattern


def build_description_variant_prompt(
 pattern: str | list[str],
 document_style: str,
 word_count: str = "400-800",
 guidelines: list[str] | None = None,
) -> str:
	guidelines_section = ""
	if guidelines:
		guidelines_section = (
		 "[Generation guidelines]\n"
		 "<guidelines>\n"
		 f"{format_pattern(guidelines)}\n"
		 "</guidelines>\n\n"
		)
	return DESCRIPTION_VARIANT_PROMPT.format(
	 pattern=format_pattern(pattern),
	 document_style=document_style,
	 word_count=word_count,
	 guidelines_section=guidelines_section,
	)


