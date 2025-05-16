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


def main():
    # API 키 로드
    load_dotenv(dotenv_path=".env")
    api_key = os.getenv("UPSTAGE_API_KEY")
    if not api_key:
        raise ValueError("API key not found in environment variables")

    # 설정
    base_config = ExperimentConfig(template_name='basic')
    train = pd.read_csv(os.path.join(base_config.data_dir, 'train.csv'))
    test = pd.read_csv(os.path.join(base_config.data_dir, 'test.csv'))

    # 토이 데이터 분할
    toy_data = train.sample(n=base_config.toy_size, random_state=base_config.random_seed).reset_index(drop=True)
    train_data, valid_data = train_test_split(
        toy_data,
        test_size=base_config.test_size,
        random_state=base_config.random_seed
    )

    # 모든 템플릿 실험
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
    config = ExperimentConfig(
        template_name=best_template,
        temperature=0.0,
        batch_size=5,
        experiment_name="final_submission"
    )
    runner = BatchExperimentRunner(config, api_key)
    raw_result = runner.run(test)
    raw_result['cor_sentence'] = raw_result['cor_sentence'].astype(str).apply(clean_output)

    # ID 기준 merge
    test_results = test[['id']].merge(raw_result, on='id', how='left')

    # <<EMPTY>> 문장만 재추론
    empty_ids = test_results[test_results['cor_sentence'] == '<<EMPTY>>']['id']
    print(f"1차 추론에서 <<EMPTY>> 문장 수: {len(empty_ids)}개")

    if not empty_ids.empty:
        retry_test = test[test['id'].isin(empty_ids)].reset_index(drop=True)
        retry_result = runner.run(retry_test)
        retry_result['cor_sentence'] = retry_result['cor_sentence'].astype(str).apply(clean_output)

        # 2차 결과 merge
        for i, row in retry_result.iterrows():
            tid = retry_test.loc[i, 'id']
            test_results.loc[test_results['id'] == tid, 'cor_sentence'] = row['cor_sentence']

    # === 3차 교정 (단순 템플릿 사용) ===
    final_empty_ids = test_results[test_results['cor_sentence'] == '<<EMPTY>>']['id']
    print(f"2차 이후에도 남은 <<EMPTY>> 문장 수: {len(final_empty_ids)}개")

    if not final_empty_ids.empty:
        # 간단한 템플릿을 사용하는 3차 config 생성
        fallback_config = ExperimentConfig(
            template_name="simple_fallback",  # 위에서 등록한 간단 템플릿
            temperature=0.0,
            batch_size=5,
            experiment_name="fallback_correction"
        )
        fallback_runner = BatchExperimentRunner(fallback_config, api_key)

        fallback_test = test[test['id'].isin(final_empty_ids)].reset_index(drop=True)
        fallback_result = fallback_runner.run(fallback_test)
        fallback_result['cor_sentence'] = fallback_result['cor_sentence'].astype(str).apply(clean_output)

        # ID 기준 덮어쓰기
        for i, row in fallback_result.iterrows():
            tid = fallback_test.loc[i, 'id']
            test_results.loc[test_results['id'] == tid, 'cor_sentence'] = row['cor_sentence']
    
    # 저장
    test_results = test_results.sort_values('id').reset_index(drop=True)
    test_results.to_csv("submission_multi_turn.csv", index=False)
    print(f"\n✅ 최종 제출 파일 생성 완료: submission_multi_turn.csv")
    print(f"예측된 문장 수: {len(test_results)}")


if __name__ == "__main__":
    main()

