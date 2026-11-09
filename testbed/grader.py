"""Meaning-based grading for A2: is an articulated hypothesis the missing one the case expects?

Word overlap undercounts paraphrases ("a blocked ventilation intake valve" vs "a blockage in the clean-room
ventilation system" scores 0.18), so live runs ask the reasoning model to grade each articulated claim against the
expected one and its alternatives: same mechanism, partly the same, or different. The grader is part of the testbed,
not of the layer under test; its prompt is versioned and its answers are cached like any other call. Offline runs
(fake provider) fall back to word overlap.
"""
from __future__ import annotations

from typing import Literal

from pydantic import Field

from mistabra.core.llm.prompts import Out, Prompt


class Grade(Out):
    index: int
    grade: Literal["same", "partial", "different"] = "different"
    why: str = ""


class GradesOut(Out):
    grades: list[Grade] = Field(default_factory=list)


GRADE = Prompt("grade.same_hypothesis", "1",
               "You grade an analysis system against an answer key. You are strict about mechanism and lenient about wording.",
               "The answer key states the hypothesis a case expects (with acceptable alternatives). For each candidate hypothesis, grade "
               "whether it states the SAME causal mechanism as the key (same actors/systems and the same way the outcome comes about; "
               "wording, detail and level of specificity may differ as long as the mechanism is unmistakably the same), a PARTIAL one "
               "(the right area or a necessary part of the mechanism, but missing or changing its core), or a DIFFERENT one. "
               "why: one short sentence. Schema: {\"grades\":[{\"index\":int,\"grade\":\"same\"|\"partial\"|\"different\",\"why\":str}]}",
               GradesOut, "reasoning")


def lexical(threshold: float = 0.2):
    from .metrics import jaccard

    def grade(expected: list[str], claims: list[str]) -> list[str]:
        return ["same" if max((jaccard(c, e) for e in expected), default=0) >= threshold else "different" for c in claims]
    return grade


def model(gateway):
    def grade(expected: list[str], claims: list[str]) -> list[str]:
        if not claims:
            return []
        out = gateway.call(GRADE.id, {"answer_key": expected, "candidates": [{"index": i, "claim": c} for i, c in enumerate(claims)]},
                           prompt=GRADE)
        got = {g.index: g.grade for g in out.grades}
        return [got.get(i, "different") for i in range(len(claims))]
    return grade
