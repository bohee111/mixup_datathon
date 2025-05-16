# templates.py
TEMPLATES = {

"enhanced_korean_pro_v3": """
### 역할
당신은 "한국어 맞춤법 교정 전문가"입니다. 
**오직 문법 오류만 고치고, 의미나 문체는 절대 바꾸지 마세요.**

### 규칙
1. 띄어쓰기, 맞춤법, 문장 부호, 조사, 어미, 높임법 오류만 수정합니다.
2. 같은 스타일(평서/의문, 존댓말/반말)을 유지하세요.
3. 의미를 바꾸거나 문장을 고치지 말고, 틀린 부분만 고칩니다.
4. 오류가 없다면 그대로 출력합니다.
5. 결과는 한 줄 문장만 출력합니다 (따옴표 X, 해설 X).

### 예시
- 잘못: 오늘날씨가좋타 -> 오늘 날씨가 좋다.
- 잘못: 아시는분? -> 아시는 분?
- 잘못: 그는책을읽었다. -> 그는 책을 읽었다.
- 잘못: 안되요 -> 안 돼요
- 잘못: 하지말껄 -> 하지 말걸

### 작업
아래 문장을 교정하세요. 문장 번호별로 한 줄씩 출력하세요.
{text}
""",

"simple_retry_v2": """
다음 문장을 맞춤법과 띄어쓰기 중심으로 간단히 교정해주세요. 의미나 문체는 바꾸지 마세요.

문장: {text}
"""
}

# main.py
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
    load_dotenv(dotenv_path=".env")
    api_key = os.getenv("UPSTAGE_API_KEY")
    if not api_key:
        raise ValueError("API key not found in environment variables")

    base_config = ExperimentConfig(template_name='enhanced_korean_pro_v3')
    train = pd.read_csv(os.path.join(base_config.data_dir, 'train.csv'))
    test = pd.read_csv(os.path.join(base_config.data_dir, 'test.csv'))

    toy_data = train.sample(n=base_config.toy_size, random_state=base_config.random_seed).reset_index(drop=True)
    train_data, valid_data = train_test_split(
        toy_data,
        test_size=base_config.test_size,
        random_state=base_config.random_seed
    )

    results = {}
    for template_name in TEMPLATES.keys():
        config = ExperimentConfig(
            template_name=template_name,
            temperature=0.0,
            batch_size=10,
            experiment_name=f"toy_experiment_{template_name}"
        )
        runner = BatchExperimentRunner(config, api_key)
        results[template_name] = runner.run_template_experiment(train_data, valid_data)

    best_template = max(
        results.items(),
        key=lambda x: x[1]['valid_recall']['recall']
    )[0]
    print(f"\n최고 성능 템플릿: {best_template}")
    print(f"Valid Recall: {results[best_template]['valid_recall']['recall']:.2f}%")
    print(f"Valid Precision: {results[best_template]['valid_recall']['precision']:.2f}%")

    print("\n=== 테스트 데이터 예측 시작 ===")
    config = ExperimentConfig(
        template_name=best_template,
        temperature=0.0,
        batch_size=10,
        experiment_name="final_submission"
    )
    runner = BatchExperimentRunner(config, api_key)
    raw_result = runner.run(test)
    raw_result['cor_sentence'] = raw_result['cor_sentence'].astype(str).apply(clean_output)

    test_results = test[['id']].merge(raw_result, on='id', how='left')
    empty_ids = test_results[test_results['cor_sentence'] == '<<EMPTY>>']['id']
    print(f"1차 추론에서 <<EMPTY>> 문장 수: {len(empty_ids)}개")

    if not empty_ids.empty:
        retry_test = test[test['id'].isin(empty_ids)].reset_index(drop=True)
        retry_result = runner.run(retry_test)
        retry_result['cor_sentence'] = retry_result['cor_sentence'].astype(str).apply(clean_output)
        for i, row in retry_result.iterrows():
            tid = retry_test.loc[i, 'id']
            test_results.loc[test_results['id'] == tid, 'cor_sentence'] = row['cor_sentence']

    final_empty_ids = test_results[test_results['cor_sentence'] == '<<EMPTY>>']['id']
    print(f"2차 이후에도 남은 <<EMPTY>> 문장 수: {len(final_empty_ids)}개")

    if not final_empty_ids.empty:
        fallback_config = ExperimentConfig(
            template_name="simple_retry_v2",
            temperature=0.0,
            batch_size=5,
            experiment_name="fallback_correction"
        )
        fallback_runner = BatchExperimentRunner(fallback_config, api_key)
        fallback_test = test[test['id'].isin(final_empty_ids)].reset_index(drop=True)
        fallback_result = fallback_runner.run(fallback_test)
        fallback_result['cor_sentence'] = fallback_result['cor_sentence'].astype(str).apply(clean_output)
        for i, row in fallback_result.iterrows():
            tid = fallback_test.loc[i, 'id']
            test_results.loc[test_results['id'] == tid, 'cor_sentence'] = row['cor_sentence']

    test_results = test_results.sort_values('id').reset_index(drop=True)
    test_results.to_csv("submission_multi_turn.csv", index=False)
    print(f"\n✅ 최종 제출 파일 생성 완료: submission_multi_turn.csv")
    print(f"예측된 문장 수: {len(test_results)}")

if __name__ == "__main__":
    main()
