import os
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from code.config import ExperimentConfig
from code.prompts.templates import TEMPLATES
from code.utils.experiment_batch import BatchExperimentRunner

def main():
    # 1. API 키 로드
    load_dotenv(dotenv_path=".env")
    api_key = os.getenv("UPSTAGE_API_KEY")
    if not api_key:
        raise ValueError("API key not found in environment variables")

    # 2. 설정 생성
    base_config = ExperimentConfig(template_name='basic')

    # 3. 데이터 로드
    train = pd.read_csv(os.path.join(base_config.data_dir, 'train.csv'))
    test = pd.read_csv(os.path.join(base_config.data_dir, 'test.csv'))

    # 4. 토이 데이터셋 구성
    toy_data = train.sample(n=base_config.toy_size, random_state=base_config.random_seed).reset_index(drop=True)
    train_data, valid_data = train_test_split(
        toy_data,
        test_size=base_config.test_size,
        random_state=base_config.random_seed
    )

    # 5. 템플릿별 실험
    results = {}
    for template_name in TEMPLATES.keys():
        config = ExperimentConfig(
            template_name=template_name,
            temperature=0.0,
            batch_size=base_config.batch_size,
            experiment_name=f"toy_batch_experiment_{template_name}"
        )
        runner = BatchExperimentRunner(config, api_key)
        results[template_name] = runner.run_template_experiment(train_data, valid_data)

    # 6. 성능 비교
    print("\n=== 템플릿별 성능 비교 ===")
    for template_name, result in results.items():
        print(f"\n[{template_name} 템플릿]")
        print("Train Recall:", f"{result['train_recall']['recall']:.2f}%")
        print("Train Precision:", f"{result['train_recall']['precision']:.2f}%")
        print("\nValid Recall:", f"{result['valid_recall']['recall']:.2f}%")
        print("Valid Precision:", f"{result['valid_recall']['precision']:.2f}%")

    # 7. 최고 성능 템플릿 선택
    best_template = max(results.items(), key=lambda x: x[1]['valid_recall']['recall'])[0]
    print(f"\n최고 성능 템플릿: {best_template}")
    print(f"Valid Recall: {results[best_template]['valid_recall']['recall']:.2f}%")
    print(f"Valid Precision: {results[best_template]['valid_recall']['precision']:.2f}%")

    # 8. 테스트 데이터 예측
    print("\n=== 테스트 데이터 예측 시작 ===")
    config = ExperimentConfig(
        template_name=best_template,
        temperature=0.0,
        batch_size=base_config.batch_size,
        experiment_name="final_batch_submission"
    )
    runner = BatchExperimentRunner(config, api_key)
    test_results = runner.run(test)

    # 9. 제출 파일 생성
    output = pd.DataFrame({
        'id': test['id'],
        'cor_sentence': test_results['cor_sentence']
    })

    output.to_csv("submission_batch.csv", index=False)
    print("\n제출 파일이 생성되었습니다: submission_batch.csv")
    print(f"사용된 템플릿: {best_template}")
    print(f"예측된 샘플 수: {len(output)}")

if __name__ == "__main__":
    main()
