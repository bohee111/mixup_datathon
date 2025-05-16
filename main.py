import os 
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from code.config import ExperimentConfig
from code.prompts.templates import TEMPLATES
from code.utils.experiment_batch import BatchExperimentRunner
from code.utils.metrics import evaluate_correction

def clean_output(text):
    if not isinstance(text, str):
        return "<<EMPTY>>"
    text = text.strip()
    if text.upper() == "<<EMPTY>>" or text == "":
        return "<<EMPTY>>"
    return text.split(":", 1)[-1].strip() if ":" in text else text

def apply_runner(test_subset, template_name, experiment_name, api_key):
    config = ExperimentConfig(
        template_name=template_name,
        temperature=0.0,
        batch_size=5,
        experiment_name=experiment_name
    )
    runner = BatchExperimentRunner(config, api_key)
    results = runner.run(test_subset)
    results['cor_sentence'] = results['cor_sentence'].astype(str).apply(clean_output)
    return results

def merge_results(original_df, new_df):
    return original_df[['id']].merge(new_df, on='id', how='left')

def overwrite_results(base_df, overwrite_df):
    for i, row in overwrite_df.iterrows():
        tid = overwrite_df.loc[i, 'id']
        base_df.loc[base_df['id'] == tid, 'cor_sentence'] = row['cor_sentence']
    return base_df

def main():
    load_dotenv(dotenv_path=".env")
    api_key = os.getenv("UPSTAGE_API_KEY")
    if not api_key:
        raise ValueError("API key not found in environment variables")

    base_config = ExperimentConfig(template_name='strict_template')
    train = pd.read_csv(os.path.join(base_config.data_dir, 'train.csv'))
    test = pd.read_csv(os.path.join(base_config.data_dir, 'test.csv'))

    # 검증용 데이터 분할
    toy_data = train.sample(n=base_config.toy_size, random_state=base_config.random_seed).reset_index(drop=True)
    train_data, valid_data = train_test_split(
        toy_data,
        test_size=base_config.test_size,
        random_state=base_config.random_seed
    )

    # 템플릿별 검증 실험
    results = {}
    for template_name in TEMPLATES.keys():
        config = ExperimentConfig(
            template_name=template_name,
            temperature=0.0,
            batch_size=5,
            experiment_name=f"toy_experiment_{template_name}"
        )
        runner = BatchExperimentRunner(config, api_key)
        results[template_name] = runner.run_template_experiment(train_data, valid_data)

    # 최고 템플릿 선택
    best_template = max(
        results.items(),
        key=lambda x: x[1]['valid_recall']['recall']
    )[0]
    print(f"\n최고 성능 템플릿: {best_template}")
    print(f"Valid Recall: {results[best_template]['valid_recall']['recall']:.2f}%")
    print(f"Valid Precision: {results[best_template]['valid_recall']['precision']:.2f}%")

    # 테스트 예측 실행
    print("\n=== 테스트 데이터 예측 시작 ===")
    test_result = apply_runner(test, best_template, "final_submission", api_key)
    test_results = merge_results(test, test_result)

    # 2차 교정
    empty_ids_1 = test_results[test_results['cor_sentence'] == '<<EMPTY>>']['id']
    if not empty_ids_1.empty:
        retry_test_2 = test[test['id'].isin(empty_ids_1)].reset_index(drop=True)
        second_result = apply_runner(retry_test_2, "relaxed_template", "retry_2", api_key)
        test_results = overwrite_results(test_results, second_result)

    # 3차 교정
    empty_ids_2 = test_results[test_results['cor_sentence'] == '<<EMPTY>>']['id']
    if not empty_ids_2.empty:
        retry_test_3 = test[test['id'].isin(empty_ids_2)].reset_index(drop=True)
        third_result = apply_runner(retry_test_3, "simple_fallback", "retry_3", api_key)
        test_results = overwrite_results(test_results, third_result)

    # 저장
    test_results = test_results.sort_values('id').reset_index(drop=True)
    test_results.to_csv("submission_multi_turn.csv", index=False)
    print(f"\n✅ 최종 제출 파일 생성 완료: submission_multi_turn.csv")
    print(f"예측된 문장 수: {len(test_results)}")

    # 검증 성능 출력
    print("\n=== 검증 데이터 최종 성능 평가 ===")
    valid_pred = apply_runner(valid_data, best_template, "final_valid_eval", api_key)
    valid_eval = evaluate_correction(valid_data, valid_pred)
    print(f"최종 검증 Recall: {valid_eval['recall']:.2f}%")
    print(f"최종 검증 Precision: {valid_eval['precision']:.2f}%")

if __name__ == "__main__":
    main()
