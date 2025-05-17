def clean_prefix(text: str) -> str:
    # '교정:', '잘못:' 등의 말머리를 제거하고 strip

def clean_column(df: pd.DataFrame, column: str = "cor_sentence") -> pd.DataFrame:
    # DataFrame의 지정 열에 대해 clean_prefix 적용
