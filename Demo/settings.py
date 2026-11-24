from pathlib import Path
import os
import torch

# Gốc project = thư mục chứa file này, trừ khi có biến môi trường.
PROJECT_ROOT = Path(os.environ.get('Regulation_QA_System', Path(__file__).resolve().parent))

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
CONFIG_DIR = PROJECT_ROOT / "config"
CACHE_DIR = PROJECT_ROOT / "cache"
REFERENCE_DIR = DATA_DIR / "references"

CORPUS_PATH = DATA_DIR / "corpus.parquet"
QA_MASTER_PATH = DATA_DIR / "qa_master.parquet"     

GENERATOR_RUN_NAME = "Qwen3-0.6B"

GENERATOR_MODEL_DIR = MODELS_DIR / "generator" / GENERATOR_RUN_NAME
DENSE_MODEL_DIR = MODELS_DIR / "retriever" / "multilingual-e5-large"

RETRIEVER_SERVING_PATH = CONFIG_DIR / "retriever_serving.json"

TOP_K = 1          # số điều luật vào generator
DISPLAY_K = 5      # số điều luật hiện làm trích dẫn

# Chạy verify_local.py --calibrate để lấy số này rồi điền vào.
ABSTAIN_THRESHOLD = None

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

EVALUATION_SPLIT = 'test'


def describe() -> str:
  lines = [f"PROJECT_ROOT : {PROJECT_ROOT}", f"DEVICE       : {DEVICE}"]

  required = {
      'corpus': CORPUS_PATH,
      'serving config': RETRIEVER_SERVING_PATH,
      'generator': GENERATOR_MODEL_DIR / 'prompt_config.json',
      'dense model': DENSE_MODEL_DIR,
  }

  for label, path in required.items():
    lines.append(f"{'OK  ' if path.exists() else 'THIẾU'} {label:15s} {path}")

  return "\n".join(lines)


if __name__ == '__main__':
  print(describe())
