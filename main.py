import os, pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from code.config import ExperimentConfig
from code.prompts.templates import TEMPLATES
from code.utils.experiment_multiturn_batch import MultiTurnBatchRunner
from code.utils.metrics import evaluate_correction

# ---------------- main ----------------
def main():
    # ① API key
    load_dotenv()
    api_key = os.getenv("UPSTAGE_API_KEY")
    if not api_key:
        raise ValueError("UPSTAGE_API_KEY not set")

    # ② 데이터
    cfg0       = ExperimentConfig(template_name="basic")
    train_df   = pd.read_csv(os.path.join(cfg0.data_dir, "train.csv"))
    test_df    = pd.read_csv(os.path.join(cfg0.data_dir, "test.csv"))
    toy_df     = train_df.sample(n=cfg0.toy_size, random_state=cfg0.random_seed)
    tr, val    = train_test_split(toy_df, test_size=cfg0.test_size,
                                  random_state=cfg0.random_seed)

    # ③ 각 템플릿 1차 → few-shot → 2차
    results = {}
    for tpl_name, tpl_text in TEMPLATES.items():
        cfg = ExperimentConfig(template_name=tpl_name, temperature=0.05)
        runner = MultiTurnBatchRunner(cfg, api_key)

        # ---- 1차 pass ----
        tr_pred1 = runner.run(tr)
        vl_pred1 = runner.run(val)

        # ★ 인덱스 맞추기 ― 여기 두 줄 추가
        val_reset = val.reset_index(drop=True)
        vl_pred1  = vl_pred1.reset_index(drop=True)

        #tr_pred1["cor_sentence"] = tr_pred1["cor_sentence"].apply(lambda s: s.split(":", 1)[-1].strip())
        #vl_pred1["cor_sentence"] = vl_pred1["cor_sentence"].apply(lambda s: s.split(":", 1)[-1].strip())

        score1 = evaluate_correction(tr, tr_pred1)
        vscore1 = evaluate_correction(val, vl_pred1)

        # ---- 2차용 few-shot: 1차에서 틀린 validation 30문장 ----
        diff  = vl_pred1["cor_sentence"] != val_reset["cor_sentence"]
        hard  = val_reset[diff].head(300)
        fewshot = ""
        for e, c in zip(hard["err_sentence"], hard["cor_sentence"]):
            fewshot += f"잘못: {e}\n교정: {c}\n\n"

        # ---- 2차 pass ----
        vl_pred2 = runner.run(val, fewshot).reset_index(drop=True)
        #vl_pred2["cor_sentence"] = vl_pred2["cor_sentence"].apply(lambda s: s.split(":", 1)[-1].strip())
        vscore2  = evaluate_correction(val_reset, vl_pred2)

        results[tpl_name] = {
            "score1": vscore1,          # 1차 검증 점수
            "score2": vscore2,          # 2차(개선) 검증 점수
            "fewshot": fewshot,
        }

    # ④ 최고 템플릿 선택(2차 점수 기준)
    best_tpl = max(results, key=lambda k: results[k]["score2"]["recall"])
    best_fs  = results[best_tpl]["fewshot"]

    print("\n=== 최고 템플릿 ===")
    print("name :", best_tpl)
    print("Recall / Precision :", 
          f"{results[best_tpl]['score2']['recall']:.4f} / {results[best_tpl]['score2']['precision']:.4f}")

    # # ⑤ 최종 테스트 예측 (멀티턴 + few-shot)
    cfg_best = ExperimentConfig(template_name=best_tpl, temperature=0.05,
                                experiment_name="final_submission")
    test_runner = MultiTurnBatchRunner(cfg_best, api_key)
    test_pred   = test_runner.run(test_df, best_fs)
    #test_pred["cor_sentence"] = test_pred["cor_sentence"].apply(lambda s: s.split(":", 1)[-1].strip())

    # ⑥ 저장
    out = pd.DataFrame({"id": test_df["id"], "cor_sentence": test_pred["cor_sentence"]})
    out.to_csv("submission_multiturn.csv", index=False)
    print("\nsubmission_multiturn.csv 생성 — rows:", len(out))

# --------------------------------------
if __name__ == "__main__":
    main()
