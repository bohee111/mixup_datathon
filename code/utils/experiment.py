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

    def _make_prompt(self, row: pd.Series) -> str:
        if self.config.template_name == 'meta_auto':
            return self.template.format(
                text=row['err_sentence'],
                age=row.get('age', ''),
                source=row.get('source', ''),
                gender=row.get('gender', '')
            )
        else:
            return self.template.format(text=row['err_sentence'])

    def _call_api_single(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        messages = [
            {
                "role": "system",
                "content": (
                    "너는 문장의 맞춤법, 띄어쓰기, 문장 부호 오류만 판단하고 최소한으로 수정하는 교정 전문가야. "
                    "의미나 말투, 문장 구조가 자연스럽다면 그대로 유지해야 해."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        data = {
            "model": self.model,
            "messages": messages
        }

        response = requests.post(self.api_url, headers=headers, json=data)
        response.raise_for_status()
        results = response.json()
        return results["choices"][0]["message"]["content"]

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        results = []
        for _, row in tqdm(data.iterrows(), total=len(data)):
            prompt = self._make_prompt(row)
            corrected = self._call_api_single(prompt)
            results.append({
                'id': row['id'],
                'cor_sentence': corrected
            })
        return pd.DataFrame(results)

    def run_template_experiment(self, train_data: pd.DataFrame, valid_data: pd.DataFrame) -> Dict:
        print(f"\n=== {self.config.template_name} 템플릿 실험 ===")

        print("\n[학습 데이터 실험]")
        train_results = self.run(train_data)
        train_recall = evaluate_correction(train_data, train_results)

        print("\n[검증 데이터 실험]")
        valid_results = self.run(valid_data)
        valid_recall = evaluate_correction(valid_data, valid_results)

        return {
            'train_recall': train_recall,
            'valid_recall': valid_recall,
            'train_results': train_results,
            'valid_results': valid_results
        }
