# code/utils/experiment_multistep.py
import requests
import pandas as pd
from tqdm import tqdm
from code.prompts.templates import MULTISTEP_TEMPLATES

class MultiStepExperimentRunner:
    def __init__(self, config, api_key):
        self.config = config
        self.api_key = api_key
        self.api_url = config.api_url
        self.model = config.model
        self.templates = config.template_names

    def _call_api(self, prompt):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}]
        }
        response = requests.post(self.api_url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']

    def _apply_templates(self, sentence):
        result = sentence
        for name in self.templates:
            prompt = MULTISTEP_TEMPLATES[name].format(text=result)
            result = self._call_api(prompt).strip()
        return result

    def run_multistep(self, data):
        results = []
        for _, row in tqdm(data.iterrows(), total=len(data)):
            corrected = self._apply_templates(row['err_sentence'])
            results.append({'id': row['id'], 'cor_sentence': corrected})
        return pd.DataFrame(results)

