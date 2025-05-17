"""
BatchReviewExperimentRunner
---------------------------
1) 12문장씩 batch로 1차 교정
2) 1차 결과를 다시 batch로 보내 남은 오류 재교정
   (오류 없으면 그대로 반환)
※ per-sentence 최대 2 call  → 규칙 “3회 이내” 준수
"""
import re, time, random, requests, pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import List, Dict
from code.utils.experiment import ExperimentRunner

# ===== 파라미터 =====
BATCH_SIZE  = 12
MAX_WORKERS = 8        # 동시 배치
QPM_LIMIT   = 180      # 계정 허용
MAX_RETRY   = 4
MAX_PASS    = 2        # 1차+2차(검수) = 2회
# =====================

_interval, _last = 60 / QPM_LIMIT, 0.0
def rate_post(url, **kw):
    global _last
    now = time.time()
    if now - _last < _interval:
        time.sleep(_interval - (now - _last))
    _last = time.time()
    return requests.post(url, timeout=60, **kw)

NUM = ["①","②","③","④","⑤","⑥","⑦","⑧","⑨","⑩","⑪","⑫"]

class BatchReviewExperimentRunner(ExperimentRunner):
    # ---------- 공통 ----------
    def _build_prompt(self, lines: List[str]) -> str:
        numbered = "\n".join(f"{NUM[i]} 잘못: {ln}" for i, ln in enumerate(lines))
        return (
            self.template.split("### 작업")[0] +
            "\n\n### 작업\n" + numbered + "\n\n교정:"
        )

    def _parse(self, text: str, k: int) -> List[str]:
        outs = [""] * k
        for ln in text.splitlines():
            m = re.match(r"^[①-⑫]\s*(.+)$", ln.strip())
            if m:
                idx = NUM.index(ln[0]); outs[idx] = m.group(1).strip()
        for i, s in enumerate(outs):
            if not s: outs[i] = "<<EMPTY>>"
        return outs
    # ---------------------------

    def _call_once(self, lines: List[str]) -> List[str]:
        hdr = {"Authorization": f"Bearer {self.api_key}",
               "Content-Type": "application/json"}
        prompt = self._build_prompt(lines)
        payload = {"model": self.model,
                   "messages":[{"role":"user","content":prompt}],
                   "temperature": self.config.temperature,
                   "max_tokens":512,"top_p":0.9,"stop":[]}

        wait = 1.0
        for _ in range(MAX_RETRY):
            r = rate_post(self.api_url, headers=hdr, json=payload)
            if r.status_code == 200:
                return self._parse(r.json()["choices"][0]["message"]["content"], len(lines))
            if r.status_code == 429:
                time.sleep(wait + random.random()); wait *= 2; continue
            r.raise_for_status()
        return lines  # 실패 시 그대로

    def _handle_batch(self, batch_df: pd.DataFrame) -> List[str]:
        # 1차 교정
        first = self._call_once(batch_df["err_sentence"].tolist())
        # 2차 검수
        second = self._call_once(first)
        return second

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.reset_index()
        fixed = [""] * len(data)

        batches = [data.iloc[i:i+BATCH_SIZE]
                   for i in range(0, len(data), BATCH_SIZE)]
        with ThreadPoolExecutor(MAX_WORKERS) as pool, \
             tqdm(total=len(batches), desc="Batch-Review", ncols=80) as bar:

            futs = {pool.submit(self._handle_batch, b): b for b in batches}
            for fut in as_completed(futs):
                batch = futs[fut]
                for orig_idx, sent in zip(batch["index"], fut.result()):
                    fixed[orig_idx] = sent
                bar.update(1)

        return pd.DataFrame({"id": data["id"], "cor_sentence": fixed})
