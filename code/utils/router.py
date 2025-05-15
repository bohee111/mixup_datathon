def select_template_by_meta(source: str, gender: str, age: int) -> str:
    if source == "질문게시판":
        return "formal"
    if int(age) <= 19:
        return "grammar_hint"
    if gender == "F":
        return "neutral"
    return "neutral"
