from __future__ import annotations

from typing import Iterable


RAG_PROMPT = (
    "You are a helpful assistant, below is a query from a user and some relevant contexts. "
    "Answer the question given the information in those contexts. Your answer should be short and concise. "
    "If you cannot find the answer to the question, just say \"I don't know\". "
    "\n\nContexts: [context] \n\nQuery: [question] \n\nAnswer:"
)

NO_DOCUMENT_PROMPT = (
    "You are a helpful assistant, below is a query from a user. "
    "Answer the question based on your knowledge. Your answer should be short and concise. "
    "\n\nQuery: [question] \n\nAnswer:"
)

REWRITE_INSTRUCTION = (
    "As an expert copy-editor, please paraphrase the input while preserving its original meaning, answer target, and scope.\n\n"
    "Requirements:\n"
    "- Preserve the original sentence type.\n"
    "- Preserve the same answer and the same level of specificity.\n"
    "- Preserve all key constraints, including time, entities, comparisons, quantities, and answer type.\n"
    "- Keep named entities, numbers, dates, and technical terms unchanged.\n"
    "- Keep roughly the same length.\n\n"
    "Output only the rewritten question."
)

OPEN_QA_ANSWER_JUDGE_PROMPT = """You are a strict judge for short-answer RAG evaluation.

Task:
Given the expected answer, its type, and the RAG response, decide whether the response correctly answers the question.

Judging rules:
- For numeric, year, date, percentage, ratio, and threshold answers, prefer exact match after simple normalization.
- For person, entity, location, method, biomarker, and technical term answers, allow semantically equivalent wording, standard aliases, inflectional variants, unit-symbol variants, and concise paraphrases if they clearly refer to the same answer.
- Prefer semantic consistency over surface-form identity. Exact wording is not required when the response expresses the same answer.
- If the expected answer appears as the core answer span inside a slightly longer response, count it as matched when the extra words only add harmless detail.
- Example: expected `Seventy-five`, response `Seventy-five marathon runners` should be judged as matched.
- For mean±variance or mean±standard-deviation answers, answering with just the central value should be judged as matched when the unit and quantity are consistent.
- Example: expected `5.4±2.2 days`, response `5.4 days` should be judged as matched.
- Accept descriptive paraphrases when they state the same fact, but reject answers that broaden, narrow, or switch to a different sibling example, subtype, or coordinate item.
- Topical similarity alone is not enough.
- Ignore case differences and minor morphology changes.

Output JSON only:
{{
  "expected_answer": "...",
  "matched": true,
  "evidence": "...",
  "normalized_expected_answer": "...",
  "normalized_response": "..."
}}

Expected answer:
{expected_answer}

Answer type:
{answer_type}

RAG response:
{response}
"""


def rag_prompt(question: str, contexts: Iterable[str]) -> str:
    return RAG_PROMPT.replace("[question]", question).replace("[context]", "\n".join(contexts))


def no_document_prompt(question: str) -> str:
    return NO_DOCUMENT_PROMPT.replace("[question]", question)


def sigma_judge_prompt(expected_answer: str, answer_type: str, response: str) -> str:
    return OPEN_QA_ANSWER_JUDGE_PROMPT.format(
        expected_answer=expected_answer,
        answer_type=answer_type,
        response=response,
    )


def rewrite_request(question: str) -> dict:
    return {
        "instruction": REWRITE_INSTRUCTION,
        "message": f"<<{question.strip()}>>",
    }


def dcmi_perturb_prompt(document: str, replace_ratio: float = 0.06) -> str:
    import re

    word_count = len(re.findall(r"\b\w+(?:[-']\w+)?\b", document.strip()))
    count = max(1, round(replace_ratio * word_count))
    return (
        f"Replace {count} key adjectives or adverbs in noticeable positions with their antonyms "
        f"in the following text, ensuring the modified text remains logically correct:\n{document.strip()}\n"
        "Return only the modified text."
    )
