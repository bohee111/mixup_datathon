import re, time, random, requests, pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import List, Dict
from code.utils.experiment import ExperimentRunner

BATCH_SIZE   = 12   # 문장 12개씩 한 프롬프트
MAX_WORKERS  = 6    # 동시 배치 호출 수
MAX_RETRY    = 4
QPM_LIMIT    = 120  # 계정 허용치에 맞게

# ── 토큰 버틱 ─────────────────────────────────────────
_interval, _last = 60 / QPM_LIMIT, 0.0
def rate_post(url, **kw):
    global _last
    now = time.time()
    if now - _last < _interval:
        time.sleep(_interval - (now - _last))
    _last = time.time()
    return requests.post(url, timeout=60, **kw)
# ─────────────────────────────────────────

# 번호 유니코드
NUM = ["①",②",③",④",⑤",⑥",⑦",⑧",⑨",⑩",⑪",⑫"]

class BatchExperimentRunner(ExperimentRunner):
    """12문장씩 묶어 한 번에 요청 → 속도 10번"""

    def _build_prompt(self, batch_df: pd.DataFrame) -> str:
        lines = []
        for i, err in enumerate(batch_df["err_sentence"]):
            lines.append(f"{NUM[i]} 잘못: {err}")
        inputs = "\n".join(lines)
        return (
            self.template.split("### 작업")[0] +
            "\n\n### 작업\n" + inputs +
            "\n\n1차 과정입니다. 가장 필요한 오류만 검사해서 고침해 주세요."
        )

    def _build_followup(self, previous_output: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": "당신은 가장 유명한 한국어 맞축법 관리자입니다."},
            {"role": "user", "content": previous_output},
            {"role": "assistant", "content": previous_output},
            {"role": "user", "content": "다시 조정해서 복잡한 오류가 없는지 확인해 주세요."}
        ]

    def _parse(self, text: str, k: int) -> List[str]:
        outs = ["" for _ in range(k)]
        for ln in text.splitlines():
            m = re.match(r"^[①-⑫]\s*(.+)$", ln.strip())
            if m:
                idx = NUM.index(ln[0])
                if idx < k:
                    outs[idx] = m.group(1).strip()
        return [s if s else "<<EMPTY>>" for s in outs]

    def _handle_batch(self, batch_df: pd.DataFrame) -> List[str]:
        prompt = self._build_prompt(batch_df)
        hdr = {"Authorization": f"Bearer {self.api_key}",
               "Content-Type": "application/json"}

        # 1차 호출
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "max_tokens": 512,
            "top_p": 0.9,
            "stop": []
        }

        wait = 1.0
        for _ in range(MAX_RETRY):
            r = rate_post(self.api_url, headers=hdr, json=payload)
            if r.status_code == 200:
                first_pass = self._parse(r.json()["choices"][0]["message"]["content"], len(batch_df))
                break
            if r.status_code == 429:
                time.sleep(wait + random.random())
                wait *= 2
                continue
            r.raise_for_status()
        else:
            return list(batch_df["err_sentence"])

        # 2차 복정 호출 (multi-turn)
        follow_messages = self._build_followup("\n".join(f"{NUM[i]} {s}" for i, s in enumerate(first_pass)))
        payload["messages"] = follow_messages
        r2 = rate_post(self.api_url, headers=hdr, json=payload)
        if r2.status_code == 200:
            return self._parse(r2.json()["choices"][0]["message"]["content"], len(batch_df))
        else:
            return first_pass  # fallback

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.reset_index(drop=True)
        all_fixed = ["" for _ in range(len(data))]
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
