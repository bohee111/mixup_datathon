import os
import time
import pandas as pd
from tqdm import tqdm
from typing import Dict, List
import requests

from code.config import ExperimentConfig
from code.prompts.templates import TEMPLATES
from code.utils.metrics import evaluate_correction

class ExperimentRunner:
    def __init__(self, config: ExperimentConfig, api_key: str):
        self.config = config
        self.api_key = api_key
        self.template = TEMPLATES[config.template_name]
        self.api_url = config.api_url
        self.model = config.model
    
    def _make_prompt(self, text: str) -> str:
        """프롬프트 생성"""
        return self.template.format(text=text)
    
    def _call_api_single(self, err_sentence: str) -> str:
        """멀티턴 구조로 API 호출"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "당신은 한국어 맞춤법·문법 교정 전문가입니다. 사용자의 문장을 정확히 교정해 주세요."},
                {"role": "user", "content": "다음은 예시입니다.\n사용자: 오늘날씨가좋타\n전문가: 오늘 날씨가 좋다.\n사용자: 아시는분?\n전문가: 아시는 분?"},
                {"role": "user", "content": f"사용자: {err_sentence}\n전문가:"}
            ]
        }
        response = requests.post(self.api_url, headers=headers, json=data)
        response.raise_for_status()
        results = response.json()
        return results["choices"][0]["message"]["content"]


    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """데이터셋에 대한 실험 실행"""
        results = []
        for _, row in tqdm(data.iterrows(), total=len(data)):
            prompt = self._make_prompt(row['err_sentence'])
            corrected = self._call_api_single(prompt)
            results.append({
                'id': row['id'],
                'cor_sentence': corrected
            })
        return pd.DataFrame(results)

    def run_template_experiment(self, train_data: pd.DataFrame, valid_data: pd.DataFrame) -> Dict:
        """템플릿별 실험 실행"""
        print(f"\n=== {self.config.template_name} 템플릿 실험 ===")
        
        # 학습 데이터로 실험
        print("\n[학습 데이터 실험]")
        train_results = self.run(train_data)
        train_recall = evaluate_correction(train_data, train_results)
        
        # 검증 데이터로 실험
        print("\n[검증 데이터 실험]")
        valid_results = self.run(valid_data)
        valid_recall = evaluate_correction(valid_data, valid_results)
        
        return {
            'train_recall': train_recall,
            'valid_recall': valid_recall,
            'train_results': train_results,
            'valid_results': valid_results
        } 
