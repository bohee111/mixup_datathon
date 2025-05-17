import re, time, random, requests, pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import List, Dict
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

from code.utils.experiment import ExperimentRunner

BATCH_SIZE   = 12
MAX_WORKERS  = 6
MAX_RETRY    = 4
QPM_LIMIT    = 120

NUM = ["①","②","③","④","⑤","⑥","⑦","⑧","⑨","⑩","⑪","⑫"]

# ── 속도 제한용 ─────────────────
_interval, _last = 60 / QPM_LIMIT, 0.0
def rate_post(url, **kw):
    global _last
    now = time.time()
    if now - _last < _interval:
        time.sleep(_interval - (now - _last))
    _last = time.time()
    return requests.post(url, timeout=180, **kw)
# ────────────────────────────────

class BatchExperimentRunner(ExperimentRunner):
    """LangChain 기반 배치 교정"""

    def _build_prompt(self, batch_df: pd.DataFrame) -> List[Dict]:
        """LangChain ChatPromptTemplate 기반 메시지 생성"""
        numbered = "\n".join([f"{NUM[i]} {s}" for i, s in enumerate(batch_df["err_sentence"])])

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "당신은 한국어 맞춤법 교정 전문가입니다. "
                "아래 numbered 문장을 보고 각 문장의 맞춤법, 띄어쓰기, 문장 부호만 교정하세요. "
                "출력은 같은 번호 체계로 각 문장 하나씩 교정 결과만 보여주세요."
            ),
            HumanMessagePromptTemplate.from_template("{numbered_input}")
        ])

        messages = prompt.format_messages(numbered_input=numbered)
        return messages_to_dict(messages)  # ✅ 여기만 바꾸면 됩니다

    def _handle_batch(self, batch_df: pd.DataFrame) -> List[str]:
        messages = self._build_prompt(batch_df)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": messages,  # ✅ 이제 올바른 dict 형식입니다
            "temperature": self.config.temperature,
            "max_tokens": 512,
            "top_p": 0.9,
            "stop": []
        }

        wait = 1.0
        for _ in range(MAX_RETRY):
            r = rate_post(self.api_url, headers=headers, json=payload)
            if r.status_code == 200:
                return self._parse(
                    r.json()["choices"][0]["message"]["content"],
                    len(batch_df)
                )
            if r.status_code == 429:
                time.sleep(wait + random.random())
                wait *= 2
                continue
            r.raise_for_status()
        return list(batch_df["err_sentence"])

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.reset_index(drop=True)
        all_fixed = [""] * len(data)

        batches = [data.iloc[i:i+BATCH_SIZE] for i in range(0, len(data), BATCH_SIZE)]

        with ThreadPoolExecutor(MAX_WORKERS) as pool, tqdm(total=len(batches), desc="Batch", ncols=80) as bar:
            futs = {pool.submit(self._handle_batch, b): b for b in batches}
            for fut in as_completed(futs):
                batch_df = futs[fut]
                fixed_list = fut.result()
                for orig_idx, sent in zip(batch_df.index, fixed_list):
                    all_fixed[orig_idx] = sent
                bar.update(1)

        return pd.DataFrame({
            "id": data["id"],
            "cor_sentence": all_fixed
        })

