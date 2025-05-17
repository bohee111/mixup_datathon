from dataclasses import dataclass
from typing import Optional
import os

@dataclass
class ExperimentConfig:
    # === 템플릿 설정 ===
    template_name: str                         # 사용할 템플릿 이름
    temperature: float = 0.0                   # LLM temperature (0=deterministic)
    batch_size: int = 12                       # 배치 크기 (BatchExperimentRunner 기준)
    experiment_name: Optional[str] = None      # 실험 결과 저장용 이름

    # === API 설정 ===
    api_url: str = "https://api.upstage.ai/v1/chat/completions"  # Solar Pro API
    model: str = "solar-pro"                   # 모델 이름
    use_langchain: bool = True                 # LangChain 사용 여부 (확장 대비)

    # === 데이터 설정 ===
    data_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    toy_size: int = 100                        # 실험용 샘플 크기
    random_seed: int = 42                      # 랜덤 시드
    test_size: float = 0.2                     # 학습/검증 분할 비율

    def __post_init__(self):
        if self.experiment_name is None:
            self.experiment_name = f"experiment_{self.template_name}"

        # 데이터 디렉토리 존재 확인
        if not os.path.exists(self.data_dir):
            raise ValueError(f"Data directory not found: {self.data_dir}")

        # 필수 파일 존재 확인
        required_files = ['train.csv', 'test.csv']
        for file in required_files:
            file_path = os.path.join(self.data_dir, file)
            if not os.path.exists(file_path):
                raise ValueError(f"Required file not found: {file_path}")
