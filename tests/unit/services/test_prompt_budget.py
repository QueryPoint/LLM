from assistant_service.services.prompt_budget import (
    calculate_prompt_budget_warning,
    prompt_budget_exceeded,
)


def test_prompt_budget_warning_calculates_percent_and_limit_boundary() -> None:
    warning = calculate_prompt_budget_warning(
        max_prompt_chars=10,
        system_instruction="ab",
        user_prompt="cd",
        prompt_markup="ef",
        prepared_context="gh",
    )
    over_limit_warning = calculate_prompt_budget_warning(
        max_prompt_chars=10,
        user_prompt="12345678901",
    )

    assert warning == 80
    assert over_limit_warning == 110
    assert prompt_budget_exceeded(max_prompt_chars=10, user_prompt="1234567890")
