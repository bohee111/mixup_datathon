# code/utils/experiment_multiturn_batch.py
import re, time, random, requests, pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import List, Dict
from code.utils.experiment import ExperimentRunner
from requests.exceptions import ReadTimeout, ConnectionError, HTTPError
import http

BATCH = 12; WORKERS = 8; RETRY = 4; QPM = 150
_interval, _last = 60/QPM, 0.0
def rpost(url, **kw):
    global _last
    now = time.time()
    if now - _last < _interval:
        time.sleep(_interval - (now - _last))
    _last = time.time()
    return requests.post(url, timeout=150, **kw)

NUM = ["①","②","③","④","⑤","⑥","⑦","⑧","⑨","⑩","⑪","⑫"]
#_PAT = re.compile(r"^[①-⑫][\s\.:]*\s*(?:교정:)?\s*(.+)$")
_PAT = re.compile(r"^[①-⑫][\s\.:]*\s*(.+)$")

class MultiTurnBatchRunner(ExperimentRunner):
    """system → 네. → 12문장 batch"""

    # 1) 메시지 빌드
    def _build_msgs(self, batch_df: pd.DataFrame, fewshot: str = "") -> List[Dict]:
        numbered = "\n".join(f"{NUM[i]} 잘못: {txt}"
                             for i, txt in enumerate(batch_df["err_sentence"]))
        user = fewshot + numbered + "\n\n교정:"
        return [
            {"role": "system",    "content": self.template.split("### 작업")[0]},
            {"role": "assistant", "content": "네."},
            {"role": "user",      "content": user},
        ]

    # 2) 1회 호출
    def _call_once(self, msgs: list[dict], k: int) -> list[str]:
        hdr  = {"Authorization": f"Bearer {self.api_key}",
                "Content-Type":  "application/json"}
        body = {"model": self.model,
                "messages": msgs,
                "temperature": self.config.temperature,
                "max_tokens": 512}

        wait = 1.0
        for _ in range(RETRY):
            try:
                r = rpost(self.api_url, headers=hdr, json=body)
                if r.status_code == 200:
                    # ------- 기존 파싱 --------
                    outs = [""] * k
                    for ln in r.json()["choices"][0]["message"]["content"].splitlines():
                        m = _PAT.match(ln.strip())
                        if m:
                            idx = NUM.index(ln[0])
                            if idx < k:
                                outs[idx] = m.group(1).strip()
                    for i, s in enumerate(outs):
                        if not s:
                            outs[i] = "<<EMPTY>>"
                    return outs
                # 429 → 지수 백오프
                if r.status_code == 429:
                    time.sleep(wait + random.random()); wait *= 2; continue
                # 5xx → 일시적 장애로 간주하고 동일하게 재시도 ★ 추가
                if 500 <= r.status_code < 600:
                    time.sleep(wait + random.random()); wait *= 2; continue
                r.raise_for_status()

            except (ReadTimeout, ConnectionError):
                time.sleep(wait + random.random()); wait *= 2; continue
            except HTTPError as e:
                # 혹시 남은 5xx HTTPError 객체도 여기서 처리
                if e.response is not None and 500 <= e.response.status_code < 600:
                    time.sleep(wait + random.random()); wait *= 2; continue
                raise

        # 재시도 초과 → 원문 반환(손실 최소화)
        return ["<<EMPTY>>"] * k

    # 3) 배치 처리 (multi-turn = system+assistant+user 한 번)
    def _handle_batch(self, batch_df: pd.DataFrame, fewshot="") -> List[str]:
        msgs = self._build_msgs(batch_df, fewshot)
        return self._call_once(msgs, len(batch_df))

    # 4) run (single pass, 외부에서 두 번 호출)
    def run(self, data: pd.DataFrame, fewshot: str = "") -> pd.DataFrame:
        data = data.reset_index(drop=True)
        fixed = [""] * len(data)

        batches = [data.iloc[i:i+BATCH] for i in range(0, len(data), BATCH)]
        with ThreadPoolExecutor(WORKERS) as pool, \
             tqdm(total=len(batches), desc="MultiTurn", ncols=80) as bar:

            fs = {pool.submit(self._handle_batch, b, fewshot): b for b in batches}
            for fut in as_completed(fs):
                bdf, out = fs[fut], fut.result()
                for i, s in zip(bdf.index, out):
                    fixed[i] = s if s != "<<EMPTY>>" else bdf.loc[i, "err_sentence"]
                bar.update(1)
        return pd.DataFrame({"id": data["id"], "cor_sentence": fixed})
