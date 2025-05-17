import os, pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from code.config import ExperimentConfig
from code.prompts.templates import TEMPLATES
from code.utils.experiment_multiturn_batch import MultiTurnBatchRunner
from code.utils.metrics import evaluate_correction

def main():
    # ① API 키
    load_dotenv()
    api_key = os.getenv("UPSTAGE_API_KEY")
    if not api_key:
        raise ValueError("UPSTAGE_API_KEY not set")

    # ② 데이터
    base_cfg = ExperimentConfig(template_name="batch")
    train_df = pd.read_csv(os.path.join(base_cfg.data_dir, "train.csv"))
    test_df  = pd.read_csv(os.path.join(base_cfg.data_dir, "test.csv"))
    toy_df   = train_df.sample(n=base_cfg.toy_size, random_state=base_cfg.random_seed)
    tr, val  = train_test_split(toy_df, test_size=base_cfg.test_size, random_state=base_cfg.random_seed)

    # ③ 1차 교정 (batch 템플릿)
    runner1 = MultiTurnBatchRunner(base_cfg, api_key)
    tr_pred1 = runner1.run(tr)
    val_pred1 = runner1.run(val)
    val_reset = val.reset_index(drop=True)
    val_pred1 = val_pred1.reset_index(drop=True)

    score1 = evaluate_correction(tr, tr_pred1)
    vscore1 = evaluate_correction(val_reset, val_pred1)

    # ④ 2차 교정 대상 추출 (1차 실패한 문장)
    diff = val_pred1["cor_sentence"] != val_reset["cor_sentence"]
    hard = pd.DataFrame({
        "id": val_reset[diff]["id"],
        "err_sentence": val_pred1[diff]["cor_sentence"],  # ← 1차 결과를 다시 입력값으로 사용
        "cor_sentence": val_reset[diff]["cor_sentence"]   # ← 여전히 기준은 정답
    })

    # ⑤ 2차 교정 실행 (basic 템플릿)
    cfg2 = ExperimentConfig(template_name="basic", temperature=0.0, experiment_name="second_pass")
    runner2 = MultiTurnBatchRunner(cfg2, api_key)
    pred2 = runner2.run(hard).reset_index(drop=True)

    # ⑥ 최종 결과 병합 (1차 성공 + 2차 교정)
    final_val = val_pred1.copy()
    final_val.loc[diff.values, "cor_sentence"] = pred2["cor_sentence"]

    vscore2 = evaluate_correction(val_reset, final_val)

    print("\n=== 평가 요약 ===")
    print("1차 recall / precision:", f"{vscore1['recall']:.2f} / {vscore1['precision']:.2f}")
    print("2차 recall / precision:", f"{vscore2['recall']:.2f} / {vscore2['precision']:.2f}")

    # ⑦ 테스트 예측 (1차 + 2차)
    test_pred1 = runner1.run(test_df)
    cfg2_test = ExperimentConfig(template_name="basic", temperature=0.0, experiment_name="second_test")
    runner2_test = MultiTurnBatchRunner(cfg2_test, api_key)

    test_pred2 = runner2_test.run(test_pred1.rename(columns={"cor_sentence": "err_sentence"}))
    test_out = pd.DataFrame({
        "id": test_df["id"],
        "cor_sentence": test_pred2["cor_sentence"]
    })

    test_out.to_csv("submission_twostep.csv", index=False)
    print("\nsubmission_twostep.csv 생성 완료 — rows:", len(test_out))

if __name__ == "__main__":
    main()
