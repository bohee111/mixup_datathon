import os
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from code.config import ExperimentConfig
from code.prompts.templates import TEMPLATES
from code.utils.experiment_batch import BatchExperimentRunner

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

    toy_data = train.sample(n=base_config.toy_size, random_state=base_config.random_seed).reset_index(drop=True)
    train_data, valid_data = train_test_split(
        toy_data,
        test_size=base_config.test_size,
        random_state=base_config.random_seed
    )

    # 1차: strict 템플릿
    print("\n=== 1차 교정 시작 (strict_template) ===")
    first_result = apply_runner(test, "strict_template", "strict_run", api_key)
    test_results = merge_results(test, first_result)

    # 2차: medium 템플릿
    print("\n=== 2차 교정 시작 (medium_template) ===")
    empty_ids_1 = test_results[test_results['cor_sentence'] == '<<EMPTY>>']['id']
    if not empty_ids_1.empty:
        retry_test_2 = test[test['id'].isin(empty_ids_1)].reset_index(drop=True)
        second_result = apply_runner(retry_test_2, "medium_template", "medium_run", api_key)
        test_results = overwrite_results(test_results, second_result)

    # 3차: simple fallback 템플릿
    print("\n=== 3차 교정 시작 (simple_fallback) ===")
    empty_ids_2 = test_results[test_results['cor_sentence'] == '<<EMPTY>>']['id']
    if not empty_ids_2.empty:
        retry_test_3 = test[test['id'].isin(empty_ids_2)].reset_index(drop=True)
        third_result = apply_runner(retry_test_3, "simple_fallback", "fallback_run", api_key)
        test_results = overwrite_results(test_results, third_result)

    # 저장: 3차까지 완료된 결과
    test_results = test_results.sort_values('id').reset_index(drop=True)
    test_results.to_csv("submission_multi_turn.csv", index=False)
    print(f"\n✅ 3차까지 교정 완료: submission_multi_turn.csv (총 {len(test_results)}문장)")

    # 4차: 전체 다시 유연하게 교정 (soft polish)
    print("\n=== 4차 전체 재교정 시작 (final_soft_polish) ===")
    final_input = test_results[['id', 'cor_sentence']].rename(columns={'cor_sentence': 'err_sentence'})
    final_result = apply_runner(final_input, "final_soft_polish", "final_polish", api_key)
    final_submission = merge_results(test_results, final_result)

    # 최종 저장
    final_submission = final_submission.sort_values('id').reset_index(drop=True)
    final_submission.to_csv("submission_final_polished.csv", index=False)
    print(f"\n🎯 최종 제출 파일 저장 완료: submission_final_polished.csv (총 {len(final_submission)}문장)")

if __name__ == "__main__":
    main()
