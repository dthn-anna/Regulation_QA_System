"""verify_local.py — chứng minh bản chạy local cho ra đúng kết quả như thực nghiệm.

    python verify_local.py             # chạy toàn bộ kiểm tra
    python verify_local.py --quick     # bỏ phần sinh câu trả lời (chỉ kiểm retriever)
    python verify_local.py --calibrate # tìm ngưỡng từ chối câu ngoài phạm vi

Cần trong data/reference/ hai file sinh ra trên Colab:
    {hybrid_run}__test_ranking.parquet          — từ notebook 04
    {generator}__test_end_to_end_predictions.parquet — từ notebook 05
"""

from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd
import os
 

sys.path.append(str(Path(__file__).resolve().parent.parent))

import Demo.settings as settings
import rag_pipeline as rag

# Câu hỏi ngoài miền để hiệu chỉnh ngưỡng từ chối. Viết thêm cho sát người dùng thật.
OUT_OF_SCOPE_QUESTIONS = [
    "Hôm nay Sài Gòn trời thế nào?",
    "Cách nấu phở bò ngon nhất là gì?",
    "Đội tuyển Việt Nam đá với ai tối nay?",
    "Bitcoin đang giá bao nhiêu?",
    "Làm sao để giảm cân nhanh?",
    "Gợi ý cho tôi vài bộ phim hay.",
    "Thủ đô của Pháp là gì?",
    "Xe máy của tôi bị chết máy phải làm sao?",
    "Viết giúp tôi một bài thơ về mùa thu.",
    "Giá vàng hôm nay tăng hay giảm?",
    "Tôi nên học tiếng Nhật hay tiếng Hàn?",
    "Bệnh cảm cúm uống thuốc gì?",
]


def load_pipeline():
  print(settings.describe(), "\n")

  return rag.RAGPipeline.load(
      corpus_path=settings.CORPUS_PATH,
      retriever_serving_path=settings.RETRIEVER_SERVING_PATH,
      generator_model_dir=settings.GENERATOR_MODEL_DIR,
      dense_model_dir=settings.DENSE_MODEL_DIR,
      cache_dir=settings.CACHE_DIR,
      device=settings.DEVICE,
  )


def find_reference(pattern: str):
  matches = sorted(settings.REFERENCE_DIR.glob(pattern))

  return matches[0] if matches else None


def check_retriever(pipeline) -> None:
  """Retriever local có tái tạo đúng ranking đã lưu trên Colab không?

  Đây là phép kiểm tra quan trọng nhất: nó bắt mọi sai lệch cấu hình (độ sâu, dot product vs cosine, quên max-pool, pyvi khác phiên bản).
  """
  print("=" * 70)
  print("1. RETRIEVER")
  print("=" * 70)

  # reference_path = find_reference(f'*{settings.EVALUATION_SPLIT}_ranking.parquet')
  reference_path = find_reference(f'best_hybrid_ranking.parquet')


  if reference_path is None:
    print(f"Bỏ qua: không có file ranking trong {settings.REFERENCE_DIR}")
    return

  print(f"Đối chiếu với: {reference_path.name}")

  saved_ranking = pd.read_parquet(reference_path)

  qa_master = pd.read_parquet(settings.QA_MASTER_PATH)

  queries = (qa_master[qa_master['split'] == settings.EVALUATION_SPLIT][['query_id', 'question']].drop_duplicates('query_id').rename(columns={'question': 'query'}))

  result = rag.verify_ranking_parity(pipeline, queries, saved_ranking, top_k=10)

  print(json.dumps(result, ensure_ascii=False, indent=2))

  if result['top1_identical_ratio'] < 1.0:
    print("\nTop-1 KHÔNG khớp hoàn toàn. Nguyên nhân thường gặp, theo thứ tự:")
    print("\t- pyvi khác phiên bản  -> BM25 tách từ khác -> điểm khác")
    print("\t- normalize_embeddings sai (MarginMSE dùng dot product, KHÔNG phải cosine)")
    print("\t- quên max-pool chunk -> doc, hoặc chunk_size/overlap lệch")
    print("\t- độ sâu ứng viên khác 50 -> min-max của hybrid đổi -> alpha hết đúng")
  else:
    print("\nRetriever local khớp tuyệt đối với thực nghiệm.")


def check_generator(pipeline) -> None:
  """Sinh lại câu trả lời trên tập test và so với dự đoán đã lưu."""
  print("\n" + "=" * 70)
  print("2. GENERATOR")
  print("=" * 70)

  reference_path = find_reference('*predictions.parquet')

  if reference_path is None:
    print(f"Bỏ qua: không có file predictions trong {settings.REFERENCE_DIR}")
    return

  print(f"Đối chiếu với: {reference_path.name}")

  saved = pd.read_parquet(reference_path)

  sample = saved.head(50)

  prompts = [pipeline.generator.render_prompt(question, context) for question, context in zip(sample['source_question'], sample['source_context'])]

  predictions = pipeline.generator.generate(prompts, batch_size=1)

  exact = sum(a == b for a, b in zip(predictions, sample['prediction']))

  print(f"\nKhớp tuyệt đối: {exact}/{len(sample)}")

  if exact < len(sample):
    print("\nLệch vài mẫu là bình thường khi đổi phần cứng:")
    print(f"\t- dtype: Colab chạy bfloat16 trên GPU, máy này chạy trên {settings.DEVICE}")
    print("\t- eval trên Colab sinh theo batch 32 với padding trái, ở đây batch 1")
    print("Cả hai đều là khác biệt dấu phẩy động, không phải sai định dạng.")
    print("Nếu lệch trên ~10% số mẫu thì mới là prompt hoặc context không khớp.")

    for prediction, expected in zip(predictions, sample['prediction']):
      if prediction != expected:
        print(f"\n\tlocal : {prediction[:150]}")
        print(f"\tcolab : {expected[:150]}")
        break
  else:
    print("Generator local khớp tuyệt đối.")


def calibrate(pipeline) -> None:
  """Tìm ngưỡng điểm để từ chối câu hỏi ngoài phạm vi."""
  print("\n" + "=" * 70)
  print("3. NGƯỠNG TỪ CHỐI")
  print("=" * 70)

  qa_master = pd.read_parquet(settings.QA_MASTER_PATH)

  in_scope = (qa_master[qa_master['split'] == settings.EVALUATION_SPLIT]['question'].drop_duplicates().sample(min(200, qa_master['question'].nunique()), random_state=42).tolist())

  print(f"In-scope: {len(in_scope)} câu | Ngoài miền: {len(OUT_OF_SCOPE_QUESTIONS)} câu\n")

  for percentile in [1.0, 5.0, 10.0]:
    result = rag.calibrate_abstain_threshold(pipeline.retriever, in_scope, OUT_OF_SCOPE_QUESTIONS, percentile=percentile)

    print(f"percentile={percentile:4.1f}  ngưỡng={result['threshold']:.4f} từ chối nhầm={result['in_scope_bị_từ_chối_nhầm']:5.2f}% chặn đúng={result['out_of_scope_bị_chặn_đúng']:5.2f}%")

  print(f"\nĐiểm trung bình — in-scope: {result['in_scope_mean']:.4f}, ngoài miền: {result['out_of_scope_mean']:.4f}")
  print(result['ghi_chú'])
  print("\nChọn một ngưỡng rồi điền vào ABSTAIN_THRESHOLD trong settings.py")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--quick', action='store_true', help='bỏ phần generator')
  parser.add_argument('--calibrate', action='store_true', help='chỉ hiệu chỉnh ngưỡng')

  arguments = parser.parse_args()

  pipeline = load_pipeline()

  if arguments.calibrate:
    calibrate(pipeline)
    return

  check_retriever(pipeline)

  if not arguments.quick:
    check_generator(pipeline)


if __name__ == '__main__':
  main()
