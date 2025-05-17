import re
import pandas as pd

def clean_prefix(text: str) -> str:
    """ '교정:', '잘못:' 과 같은 말머리 제거 + strip """
    if not isinstance(text, str):
        return "<<EMPTY>>"
    text = text.strip()
    if text.upper() == "<<EMPTY>>" or text == "":
        return "<<EMPTY>>"
    return re.split(r"^\\s*(교정|잘못)\\s*[:：]\\s*", text)[-1].strip()

def clean_column(df: pd.DataFrame, column: str = "cor_sentence") -> pd.DataFrame:
    """ 특정 열에 대해 clean_prefix 적용 """
    df[column] = df[column].apply(clean_prefix)
    return df
