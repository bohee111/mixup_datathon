import os
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from code.config import ExperimentConfig
from code.prompts.templates import TEMPLATES
from code.utils.experiment import ExperimentRunner
from code.utils.metrics import evaluate_correction

def main():
    # API 키 로드
    load_dotenv(dotenv_path=".env")
    api_key = os.getenv("UPSTAGE_API_KEY")
    if not api_key:
        raise ValueError("API key not found in environment variables")
    
    # 기본 설정 생성
    template_name = 'basic'
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
    
    # 지정 템플릿으로 제출 파일 생성
    print("\n=== 테스트 데이터 예측 시작 ===")
    config = ExperimentConfig(
        template_name='basic',
        temperature=0.0,
        batch_size=5,
        experiment_name="final_submission"
    )
    
    runner = ExperimentRunner(config, api_key)
    test_results = runner.run(test)
    
    output = pd.DataFrame({
        'id': test['id'],
        'cor_sentence': test_results['cor_sentence']
    })
    
    output.to_csv("submission_trial.csv", index=False)
    print("\n제출 파일이 생성되었습니다: submission_trial.csv")
    print(f"사용된 템플릿: basic")
    print(f"예측된 샘플 수: {len(output)}")

if __name__ == "__main__":
    main()
