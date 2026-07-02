from math import floor


def calculate_prompt_budget_used_chars(
    *,
    system_instruction: str = "",
    user_prompt: str = "",
    prompt_markup: str = "",
    prepared_context: str = "",
) -> int:
    return (
        len(system_instruction)
        + len(user_prompt)
        + len(prompt_markup)
        + len(prepared_context)
    )


def calculate_prompt_budget_warning(
    *,
    max_prompt_chars: int,
    system_instruction: str = "",
    user_prompt: str = "",
    prompt_markup: str = "",
    prepared_context: str = "",
) -> int:
    _validate_max_prompt_chars(max_prompt_chars)
    used_chars = calculate_prompt_budget_used_chars(
        system_instruction=system_instruction,
        user_prompt=user_prompt,
        prompt_markup=prompt_markup,
        prepared_context=prepared_context,
    )
    return floor(used_chars / max_prompt_chars * 100)


def prompt_budget_exceeded(
    *,
    max_prompt_chars: int,
    system_instruction: str = "",
    user_prompt: str = "",
    prompt_markup: str = "",
    prepared_context: str = "",
) -> bool:
    _validate_max_prompt_chars(max_prompt_chars)
    return (
        calculate_prompt_budget_used_chars(
            system_instruction=system_instruction,
            user_prompt=user_prompt,
            prompt_markup=prompt_markup,
            prepared_context=prepared_context,
        )
        >= max_prompt_chars
    )


def _validate_max_prompt_chars(max_prompt_chars: int) -> None:
    if max_prompt_chars <= 0:
        raise ValueError("max_prompt_chars must be positive")
