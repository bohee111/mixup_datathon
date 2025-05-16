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

    # 기본 설정 생성
    base_config = ExperimentConfig(template_name='basic')

    # 데이터 로드
    train = pd.read_csv(os.path.join(base_config.data_dir, 'train.csv'))
    test = pd.read_csv(os.path.join(base_config.data_dir, 'test.csv'))

    # 토이 데이터셋 생성
    toy_data = train.sample(n=base_config.toy_size, random_state=base_config.random_seed).reset_index(drop=True)

    # train/valid 분할
    train_data, valid_data = train_test_split(
        toy_data,
        test_size=base_config.test_size,
        random_state=base_config.random_seed
    )

    # 모든 템플릿으로 실험
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

    # 최고 성능 템플릿 찾기
    best_template = max(
        results.items(),
        key=lambda x: x[1]['valid_recall']['recall']
    )[0]
    print(f"\n최고 성능 템플릿: {best_template}")
    print(f"Valid Recall: {results[best_template]['valid_recall']['recall']:.2f}%")
    print(f"Valid Precision: {results[best_template]['valid_recall']['precision']:.2f}%")

    # 최고 성능 템플릿으로 예측
    print("\n=== 테스트 데이터 예측 시작 ===")
    config = ExperimentConfig(
        template_name=best_template,
        temperature=0.0,
        batch_size=5,
        experiment_name="final_submission"
    )
    runner = BatchExperimentRunner(config, api_key)
    test_results = runner.run(test)
    test_results['cor_sentence'] = test_results['cor_sentence'].astype(str).apply(clean_output)

    # <<EMPTY>> 문장만 다시 2차 교정
    empty_idx = test_results['cor_sentence'].str.upper() == "<<EMPTY>>"
    print(f"1차 추론에서 <<EMPTY>> 문장 수: {empty_idx.sum()}개")
    if empty_idx.sum() > 0:
        retry_test = test[empty_idx].reset_index(drop=True)
        retry_results = runner.run(retry_test)
        retry_results['cor_sentence'] = retry_results['cor_sentence'].astype(str).apply(clean_output)
        test_results.loc[empty_idx, 'cor_sentence'] = retry_results['cor_sentence'].values

    # 최종 저장
    test_results['id'] = test['id'].values  # 정렬 맞춤
    test_results.to_csv("submission_multi_turn.csv", index=False)
    print(f"\n✅ 최종 제출 파일 생성 완료: submission_multi_turn.csv")
    print(f"예측된 문장 수: {len(test_results)}")


if __name__ == "__main__":
    main()
