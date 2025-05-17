# code/utils/experiment_multiturn_batch.py
import re, time, random, requests, pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import List, Dict
from code.utils.experiment import ExperimentRunner
from requests.exceptions import ReadTimeout, ConnectionError, HTTPError
from code.config import ExperimentConfig  # 추가

# 환경 설정 로드 (외부 설정값 활용)
cfg_batch = ExperimentConfig(template_name="batch")
BATCH = cfg_batch.batch_size    # e.g., 12
WORKERS = cfg_batch.workers     # e.g., 8
RETRY = 4
QPM = cfg_batch.qpm             # e.g., 150

_interval, _last = 60 / QPM, 0.0
NUM = ["①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨", "⑩", "⑪", "⑫"]
_PAT = re.compile(r"^[①-⑫][\s\.:]*\s*(.+)$")

def rpost(url, **kw):
    global _last
    now = time.time()
    if now - _last < _interval:
        time.sleep(_interval - (now - _last))
    _last = time.time()
    return requests.post(url, timeout=150, **kw)

class MultiTurnBatchRunner(ExperimentRunner):
    """system → 네. → 12문장 batch"""

    def _build_msgs(self, batch_df: pd.DataFrame, fewshot: str = "") -> List[Dict]:
        numbered = "\n".join(f"{NUM[i]} 잘못: {txt}" for i, txt in enumerate(batch_df["err_sentence"]))
        user = fewshot + numbered + "\n\n교정:"
        return [
            {"role": "system",    "content": self.template.split("### 작업")[0]},
            {"role": "assistant", "content": "네."},
            {"role": "user",      "content": user},
        ]

    def _call_once(self, msgs: list[dict], k: int) -> list[str]:
        hdr = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        body = {
            "model": self.model,
            "messages": msgs,
            "temperature": self.config.temperature,
            "max_tokens": 512
        }

        wait = 1.0
        for _ in range(RETRY):
            try:
                r = rpost(self.api_url, headers=hdr, json=body)
                if r.status_code == 200:
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

                if r.status_code == 429 or (500 <= r.status_code < 600):
                    time.sleep(wait + random.random())
                    wait *= 2
                    continue
                r.raise_for_status()

            except (ReadTimeout, ConnectionError) as e:
                print(f"[TIMEOUT] {str(e)}")
                time.sleep(wait + random.random())
                wait *= 2
                continue
            except HTTPError as e:
                if e.response is not None and 500 <= e.response.status_code < 600:
                    time.sleep(wait + random.random())
                    wait *= 2
                    continue
                raise
            except Exception as e:
                print(f"[ERROR] API 호출 실패: {str(e)}")
                break

        return ["<<EMPTY>>"] * k

    def _handle_batch(self, batch_df: pd.DataFrame, fewshot="") -> List[str]:
        msgs = self._build_msgs(batch_df, fewshot)
        return self._call_once(msgs, len(batch_df))

    def run(self, data: pd.DataFrame, fewshot: str = "") -> pd.DataFrame:
        data = data.reset_index(drop=True)
        fixed = [""] * len(data)

        batches = [data.iloc[i:i + BATCH] for i in range(0, len(data), BATCH)]
        with ThreadPoolExecutor(WORKERS) as pool, tqdm(total=len(batches), desc="MultiTurn", ncols=80) as bar:
            fs = {pool.submit(self._handle_batch, b, fewshot): b for b in batches}
            for fut in as_completed(fs):
                bdf, out = fs[fut], fut.result()
                for i, s in zip(bdf.index, out):
                    fixed[i] = s if s != "<<EMPTY>>" else f"<<SKIPPED>> {bdf.loc[i, 'err_sentence']}"
                bar.update(1)

        return pd.DataFrame({"id": data["id"], "cor_sentence": fixed})
