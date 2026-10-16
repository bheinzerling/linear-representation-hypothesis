"""Prompt families for evaluating parity behavior before causal steering."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Sequence


@dataclass(frozen=True)
class ParityPromptVariant:
    variant_id: str
    category: str
    description: str
    pattern: str
    demonstration_values: tuple[int, ...]

    def render(self, value: int) -> str:
        if self.pattern.count("{number}") != 1:
            raise ValueError(f"{self.variant_id} must have exactly one number placeholder")
        return self.pattern.format(number=int(value))


def parity_word(value: int) -> str:
    return "even" if int(value) % 2 == 0 else "odd"


def digit_coverage_demonstrations(
    train_values: Sequence[int],
    *,
    examples_per_digit: int = 1,
    minimum_value: int = 0,
) -> tuple[int, ...]:
    """Select equally many train values for every final decimal digit."""

    train_set = {int(value) for value in train_values}
    selected = []
    for digit in range(10):
        matches = sorted(
            value
            for value in train_set
            if value >= int(minimum_value) and value % 10 == digit
        )
        if len(matches) < int(examples_per_digit):
            raise ValueError(
                f"need {examples_per_digit} training values with final digit {digit} "
                f"and minimum value {minimum_value}"
            )
        selected.extend(matches[: int(examples_per_digit)])
    return tuple(selected)


def _classification_lines(values: Sequence[int]) -> str:
    return "\n".join(f"{value}: {parity_word(value)}" for value in values)


def _question_lines(values: Sequence[int]) -> str:
    return "\n".join(
        f"Question: Is {value} even or odd? Answer: {parity_word(value)}"
        for value in values
    )


def _modulo_lines(values: Sequence[int]) -> str:
    return "\n".join(
        f"{value} modulo 2 is {value % 2}, so {value} is {parity_word(value)}."
        for value in values
    )


def _suffix_lines(values: Sequence[int]) -> str:
    return "\n".join(
        f"{value} ends in {value % 10}, so {value} is {parity_word(value)}."
        for value in values
    )


def build_prompt_variants(train_values: Sequence[int]) -> tuple[ParityPromptVariant, ...]:
    """Build matched parity prompt variants using demonstrations from train values."""

    digit_demos = digit_coverage_demonstrations(train_values)
    reverse_digit_demos = tuple(reversed(digit_demos))
    three_digit_demos = digit_coverage_demonstrations(train_values, minimum_value=100)
    three_digit_twenty = digit_coverage_demonstrations(
        train_values,
        examples_per_digit=2,
        minimum_value=100,
    )
    shuffled_three_digit_demos = []
    for seed in (11, 29, 47):
        values = list(three_digit_demos)
        random.Random(seed).shuffle(values)
        shuffled_three_digit_demos.append(tuple(values))
    balanced_eight = tuple(
        value
        for pair in zip(
            sorted(value for value in train_values if value % 2 == 0)[:4],
            sorted(value for value in train_values if value % 2 == 1)[:4],
        )
        for value in pair
    )
    rule = (
        "Rule: A whole number is even when its final decimal digit is 0, 2, 4, 6, "
        "or 8. It is odd when its final decimal digit is 1, 3, 5, 7, or 9."
    )
    variants = [
        ParityPromptVariant(
            variant_id="zero_shot_direct",
            category="zero_shot",
            description="Direct question without demonstrations or an explicit rule.",
            pattern="Question: Is {number} even or odd? Answer:",
            demonstration_values=(),
        ),
        ParityPromptVariant(
            variant_id="original_two_shot",
            category="few_shot",
            description="The original evaluation's two demonstrations.",
            pattern=(
                "Question: What is the parity of 42? Answer: even\n"
                "Question: What is the parity of 17? Answer: odd\n"
                "Question: What is the parity of {number}? Answer:"
            ),
            demonstration_values=(42, 17),
        ),
        ParityPromptVariant(
            variant_id="balanced_eight_shot",
            category="few_shot",
            description="Eight alternating even/odd train-value demonstrations.",
            pattern=(
                "Classify each integer as even or odd.\n"
                f"{_classification_lines(balanced_eight)}\n"
                "{number}:"
            ),
            demonstration_values=balanced_eight,
        ),
        ParityPromptVariant(
            variant_id="digit_coverage_ascending_10shot",
            category="few_shot",
            description="One train demonstration for every final digit, ordered 0 through 9.",
            pattern=(
                "Classify each integer as even or odd.\n"
                f"{_classification_lines(digit_demos)}\n"
                "{number}:"
            ),
            demonstration_values=digit_demos,
        ),
        ParityPromptVariant(
            variant_id="digit_coverage_descending_10shot",
            category="few_shot_control",
            description="The same final-digit demonstrations in reverse order.",
            pattern=(
                "Classify each integer as even or odd.\n"
                f"{_classification_lines(reverse_digit_demos)}\n"
                "{number}:"
            ),
            demonstration_values=reverse_digit_demos,
        ),
        ParityPromptVariant(
            variant_id="explicit_rule_zero_shot",
            category="rule",
            description="Complete decimal final-digit rule without demonstrations.",
            pattern=f"{rule}\nInteger: {{number}}\nParity (answer only even or odd):",
            demonstration_values=(),
        ),
        ParityPromptVariant(
            variant_id="explicit_rule_digit_10shot",
            category="rule_and_few_shot",
            description="Complete rule plus one train demonstration for every final digit.",
            pattern=(
                f"{rule}\nExamples:\n{_question_lines(digit_demos)}\n"
                "Question: Is {number} even or odd? Answer:"
            ),
            demonstration_values=digit_demos,
        ),
        ParityPromptVariant(
            variant_id="modulo_reasoning_10shot",
            category="worked_few_shot",
            description="Worked modulo-2 demonstrations followed by an answer-only query.",
            pattern=(
                "Use division by 2 to determine parity.\n"
                f"{_modulo_lines(digit_demos)}\n"
                "Now classify {number}. Answer only even or odd:"
            ),
            demonstration_values=digit_demos,
        ),
        ParityPromptVariant(
            variant_id="three_digit_rule_10shot",
            category="length_matched_rule_and_few_shot",
            description="Complete rule plus one three-digit train example for every final digit.",
            pattern=(
                f"{rule}\nExamples:\n{_question_lines(three_digit_demos)}\n"
                "Question: Is {number} even or odd? Answer:"
            ),
            demonstration_values=three_digit_demos,
        ),
        ParityPromptVariant(
            variant_id="three_digit_suffix_10shot",
            category="length_matched_worked_few_shot",
            description="Ten three-digit examples explicitly connecting final digits to parity.",
            pattern=(
                f"{rule}\nWorked examples:\n{_suffix_lines(three_digit_demos)}\n"
                "Classify {number} using its final digit. Answer only even or odd:"
            ),
            demonstration_values=three_digit_demos,
        ),
        ParityPromptVariant(
            variant_id="three_digit_suffix_20shot",
            category="length_matched_worked_few_shot",
            description="Twenty three-digit worked examples, two for every final digit.",
            pattern=(
                f"{rule}\nWorked examples:\n{_suffix_lines(three_digit_twenty)}\n"
                "Classify {number} using its final digit. Answer only even or odd:"
            ),
            demonstration_values=three_digit_twenty,
        ),
    ]
    for index, demonstrations in enumerate(shuffled_three_digit_demos, start=1):
        variants.append(
            ParityPromptVariant(
                variant_id=f"three_digit_rule_10shot_shuffle_{index}",
                category="length_matched_rule_order_control",
                description=(
                    "The same ten three-digit rule demonstrations in a fixed shuffled order."
                ),
                pattern=(
                    f"{rule}\nExamples:\n{_question_lines(demonstrations)}\n"
                    "Question: Is {number} even or odd? Answer:"
                ),
                demonstration_values=demonstrations,
            )
        )
    return tuple(variants)
