from routemcp.evaluation import ExpectedResult, load_dataset, validate_answer


def test_evaluation_dataset_and_validators() -> None:
    cases = load_dataset("evaluation/sample.jsonl")
    assert len(cases) == 3
    assert validate_answer("The answer is 60.", ExpectedResult(type="regex", value=r"\b60\b"))
    assert validate_answer(
        '{"name": "Ada", "role": "engineer"}',
        ExpectedResult(type="json_fields", fields=["name", "role"]),
    )
    assert not validate_answer("not json", ExpectedResult(type="json_fields", fields=["name"]))
