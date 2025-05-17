# main.py (개선된 멀티턴 프롬프트 파이프라인)
import os, pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from code.config import ExperimentConfig
from code.prompts.templates import TEMPLATES
from code.utils.experiment_multiturn_batch import MultiTurnBatchRunner
from code.utils.metrics import evaluate_correction


def main():
    # ① API key 로딩
    load_dotenv()
    api_key = os.getenv("UPSTAGE_API_KEY")
    if not api_key:
        raise ValueError("UPSTAGE_API_KEY not set")

    # ② 데이터 로딩
    cfg = ExperimentConfig(template_name="batch")
    train_df = pd.read_csv(os.path.join(cfg.data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(cfg.data_dir, "test.csv"))
    toy_df = train_df.sample(n=cfg.toy_size, random_state=cfg.random_seed)
    tr, val = train_test_split(toy_df, test_size=cfg.test_size, random_state=cfg.random_seed)

    # ③ 1차 교정
    runner1 = MultiTurnBatchRunner(cfg, api_key)
    tr_pred1 = runner1.run(tr)
    vl_pred1 = runner1.run(val)

    val_reset = val.reset_index(drop=True)
    vl_pred1 = vl_pred1.reset_index(drop=True)

    score1 = evaluate_correction(val_reset, vl_pred1)

    # ④ few-shot 예시 생성 (1차에서 틀린 문장 위주)
    diff = vl_pred1["cor_sentence"] != val_reset["cor_sentence"]
    hard = val_reset[diff].head(300)
    fewshot = ""
    for e, c in zip(hard["err_sentence"], hard["cor_sentence"]):
        fewshot += f"잘못: {e}\n교정: {c}\n\n"

    # ⑤ 2차 교정 (동일 템플릿 유지 or 바꿔도 OK)
    cfg2 = ExperimentConfig(template_name="batch")  # or "batch+"
    runner2 = MultiTurnBatchRunner(cfg2, api_key)
    vl_pred2 = runner2.run(val, fewshot=fewshot)
    vl_pred2 = vl_pred2.reset_index(drop=True)

    score2 = evaluate_correction(val_reset, vl_pred2)

    print("\n=== 최종 평가 결과 (2차 기준) ===")
    print(f"Recall / Precision: {score2['recall']:.4f} / {score2['precision']:.4f}")

    # ⑥ 최종 예측 수행 (테스트셋)
    test_runner = MultiTurnBatchRunner(cfg2, api_key)
    test_pred = test_runner.run(test_df, fewshot=fewshot)
    out = pd.DataFrame({
        "id": test_df["id"],
        "cor_sentence": test_pred["cor_sentence"]
    })
    out.to_csv("submission_multiturn.csv", index=False)
    print("\nsubmission_multiturn.csv 생성 완료 — rows:", len(out))


if __name__ == "__main__":
    main()
