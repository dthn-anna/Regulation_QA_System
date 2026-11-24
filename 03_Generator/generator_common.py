from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import math
import re

import numpy as np
import pandas as pd

MODULE_VERSION = 5
EMPTY_TARGET_VALUES = {'', 'nan', 'none', 'null', '<na>'}

def require_version(minimum_version: int) -> None:
  if MODULE_VERSION < minimum_version:
    raise ImportError(
      f"generator_common ver {MODULE_VERSION}, cần >= {minimum_version}.\n"
      f"Nếu đã ghi đè file trên Drive, module cũ vẫn còn trong bộ nhớ.\n"
      f"Chạy:\n"
      f"\timport importlib, generator_common\n"
      f"\timportlib.reload(generator_common)\n"
      f"\tgc = generator_common\n"
      f"or Runtime -> Restart session -> Rerun"
    )
  print(f"generator_common v{MODULE_VERSION} (need >= {minimum_version})")

# 1. Chuẩn bị dữ liệu
def build_generator_dataframe(qa_master: pd.DataFrame, split_name: str, corpus: pd.DataFrame, passage_column: str = 'embedding_text', target_column: str = 'abstractive answer', drop_empty_target: bool = True) -> pd.DataFrame:
  frame = qa_master[qa_master['split'] == split_name].copy()

  if passage_column not in corpus.columns:
    raise ValueError(f"corpus không có cột '{passage_column}'. Có: {list(corpus.columns)}")
 
  if target_column not in frame.columns:
    raise ValueError(f"qa_master không có cột '{target_column}'. Có: {list(frame.columns)}")
 
  text_by_doc = dict(zip(corpus['doc_id'], corpus[passage_column].astype(str)))
 
  missing = set(frame['doc_id']) - set(text_by_doc)
  if missing:
    raise ValueError(f"{len(missing)} doc_id không có trong corpus, ví dụ {sorted(missing)[:3]}")

  frame['source_question'] = frame['question'].astype(str).str.strip()
  frame['source_context'] = frame['doc_id'].map(text_by_doc)
  frame['target'] = frame[target_column].astype(str).str.strip()
 
  if drop_empty_target:
    empty_mask = frame['target'].str.lower().isin(EMPTY_TARGET_VALUES)
 
    if empty_mask.any():
      print(f"[{split_name}] loại {int(empty_mask.sum()):,} mẫu có target rỗng/'nan'.")
      frame = frame[~empty_mask]
 
  keep = ['qa_id', 'query_id', 'doc_id', 'source_question', 'source_context', 'target']
 
  return frame[keep].reset_index(drop=True)


def attach_retrieved_context(frame: pd.DataFrame, ranking: pd.DataFrame, corpus: pd.DataFrame, top_k: int = 1, passage_column: str = 'embedding_text', context_separator: str = '\n\n') -> pd.DataFrame:
  required_columns = {'query_id', 'doc_id', 'rank'}
  missing_columns = required_columns - set(ranking.columns)
 
  if missing_columns:
    raise ValueError(f"ranking thiếu cột {sorted(missing_columns)}. Có lẽ đang đọc nhầm file metrics_per_query thay vì ranking. Cột hiện có: {list(ranking.columns)}")
 
  missing_queries = set(frame['query_id']) - set(ranking['query_id'])
 
  if missing_queries:
    raise ValueError(f"{len(missing_queries)} câu hỏi không có trong ranking — sai split hoặc ranking sinh trước khi lọc rò rỉ. Ví dụ: {sorted(missing_queries)[:3]}")

  text_by_doc = dict(zip(corpus['doc_id'], corpus[passage_column].astype(str)))
 
  top_docs = (ranking[ranking['rank'] <= top_k].sort_values(['query_id', 'rank']).groupby('query_id')['doc_id'].apply(list).to_dict())
 
  retrieved_contexts = []
  retrieved_hits = []
  retrieved_ranks = []
  retrieved_doc_ids = []
 
  for query_id, gold_doc_id in zip(frame['query_id'], frame['doc_id']):
    doc_ids = top_docs.get(query_id, [])
 
    retrieved_contexts.append(context_separator.join(text_by_doc.get(doc_id, '') for doc_id in doc_ids))
    retrieved_hits.append(gold_doc_id in doc_ids)
    retrieved_doc_ids.append(list(doc_ids))
    retrieved_ranks.append(doc_ids.index(gold_doc_id) + 1 if gold_doc_id in doc_ids else -1)
 
  result = frame.copy()
 
  result['gold_context'] = result['source_context']
  result['source_context'] = retrieved_contexts
  result['retrieved_hit'] = retrieved_hits
  result['retrieved_rank'] = retrieved_ranks
  result['retrieved_doc_ids'] = retrieved_doc_ids
  result['retrieval_top_k'] = top_k    

  return result

REFUSAL_PATTERNS = [
    r'không\s+(có|đề\s*cập|nêu|quy\s*định|tìm\s*thấy|xác\s*định)',
    r'chưa\s+(có|đề\s*cập|rõ)',
    r'không\s+thuộc\s+phạm\s+vi',
    r'điều\s+luật\s+(trên|này)\s+không',
]

# Con số, ngưỡng, mã môn — thứ mà model bịa ra sẽ gây hại nhất.
SPECIFIC_CLAIM_PATTERN = r'\d+[,.]?\d*\s*(%|điểm|tín\s*chỉ|học\s*kỳ|năm|tháng|tuần|ngày)|[A-Z]{2,}\d{2,}'

def looks_like_refusal(text: str) -> bool:
  lowered = str(text).lower()
 
  return any(re.search(pattern, lowered) for pattern in REFUSAL_PATTERNS)

def has_specific_claim(text: str) -> bool:
  return bool(re.search(SPECIFIC_CLAIM_PATTERN, str(text), flags=re.IGNORECASE))


def behaviour_report(hypotheses: list[str], retrieved_hits: list[bool]) -> pd.DataFrame:
  frame = pd.DataFrame({
      'hypothesis': hypotheses,
      'retrieved_hit': list(retrieved_hits),
  })
 
  frame['refusal'] = frame['hypothesis'].map(looks_like_refusal)
  frame['specific_claim'] = frame['hypothesis'].map(has_specific_claim)
 
  records = []
 
  for label, subset in [('Retriever ĐÚNG', frame[frame['retrieved_hit']]),
                        ('Retriever SAI', frame[~frame['retrieved_hit']])]:
    if subset.empty:
      continue
 
    records.append({
        'nhóm': label,
        'n': len(subset),
        'tỷ_lệ_từ_chối': round(subset['refusal'].mean() * 100, 2),
        'tỷ_lệ_khẳng_định_cụ_thể': round(subset['specific_claim'].mean() * 100, 2),
        # Nguy hiểm nhất: context sai, không từ chối, vẫn đưa ra con số.
        'khẳng_định_mà_không_từ_chối': round(((~subset['refusal']) & subset['specific_claim']).mean() * 100, 2)
    })
 
  return pd.DataFrame(records)

# 2. Metric
def tokenize_for_metric(text:str, lowercase: bool = True, strip_punctuation: bool = True) -> list[str]:
  text = str(text)
  if lowercase: 
    text = text.lower()
  if strip_punctuation:
    text = re.sub(r"[^\w\s]", " ", text)
 
  return text.split()

def _ngram_counts(tokens: list[str], n: int) -> Counter:
  return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n+1))

def corpus_bleu(references: list[str], hypotheses: list[str], max_n: int = 4, **tokenize_kwargs) -> float:
  if len(references) != len(hypotheses):
    raise ValueError("References must have the sample length with Hypotheses.")
 
  clipped = [0] * max_n
  total = [0] * max_n
 
  hypothesis_length = 0
  reference_length = 0

  for reference, hypothesis in zip(references, hypotheses):
    reference_tokens = tokenize_for_metric(reference, **tokenize_kwargs)
    hypothesis_tokens = tokenize_for_metric(hypothesis, **tokenize_kwargs)
 
    hypothesis_length += len(hypothesis_tokens)
    reference_length += len(reference_tokens)
 
    for n in range(1, max_n + 1):
      hypothesis_ngrams = _ngram_counts(hypothesis_tokens, n)
      reference_ngrams = _ngram_counts(reference_tokens, n)
 
      total[n-1] += max(0, len(hypothesis_tokens) - n + 1)
 
      for ngram, count in hypothesis_ngrams.items():
        clipped[n-1] += min(count, reference_ngrams.get(ngram, 0))

  if hypothesis_length == 0:
    return 0.0
 
  precisions = []
 
  for n in range(max_n):
    if total[n] == 0 or clipped[n] == 0:
      return 0.0
 
    precisions.append(clipped[n] / total[n])

  # Chuẩn BLEU dùng >= : câu sinh dài bằng câu tham chiếu không bị phạt.
  brevity_penalty = (1.0 if hypothesis_length >= reference_length else math.exp(1 - reference_length / hypothesis_length))
 
  log_mean = sum(math.log(p) for p in precisions) / max_n
 
  return brevity_penalty * math.exp(log_mean)

def _lcs_length(a: list[str], b: list[str]) -> int:
  if not a or not b:
    return 0
 
  previous = [0] * (len(b) + 1)
 
  for token_a in a:
    current = [0]
 
    for index_b, token_b in enumerate(b):
      if token_a == token_b:
        current.append(previous[index_b] + 1)
      else:
        current.append(max(current[-1], previous[index_b + 1]))
 
    previous = current
 
  return previous[-1]


def rouge_l(reference: str, hypothesis: str, beta: float = 1.2, **tokenize_kwargs) -> float:
  reference_tokens = tokenize_for_metric(reference, **tokenize_kwargs)
  hypothesis_tokens = tokenize_for_metric(hypothesis, **tokenize_kwargs)
 
  if not reference_tokens or not hypothesis_tokens:
    return 0.0
 
  lcs = _lcs_length(reference_tokens, hypothesis_tokens)
 
  if lcs == 0:
    return 0.0
 
  precision = lcs / len(hypothesis_tokens)
  recall = lcs / len(reference_tokens)
 
  return ((1 + beta ** 2) * precision * recall) / (recall + beta ** 2 * precision)

def token_f1(reference: str, hypothesis: str, **tokenize_kwargs) -> float:
 
  reference_tokens = tokenize_for_metric(reference, **tokenize_kwargs)
  hypothesis_tokens = tokenize_for_metric(hypothesis, **tokenize_kwargs)
 
  common = Counter(reference_tokens) & Counter(hypothesis_tokens)
  overlap = sum(common.values())
 
  if overlap == 0:
    return 0.0
 
  precision = overlap / len(hypothesis_tokens)
  recall = overlap / len(reference_tokens)
 
  return 2 * precision * recall / (precision + recall)

# 3. Evaluation
# 3. Đánh giá
def compute_bertscore(references: list[str], hypotheses: list[str], model_type: str = None, language: str = 'vi', batch_size: int = 64) -> np.ndarray:
  from bert_score import score as bert_score_fn
 
  kwargs = dict(batch_size=batch_size, verbose=False)
 
  if model_type:
    kwargs['model_type'] = model_type
  else:
    kwargs['lang'] = language
 
  _, _, f1 = bert_score_fn(hypotheses, references, **kwargs)
 
  return f1.numpy()


def evaluate_generations(references: list[str], hypotheses: list[str], qa_ids: list[str] = None, use_bertscore: bool = True, bertscore_model: str = None, **tokenize_kwargs) -> pd.DataFrame:
  # Trả về bảng per-example. Dùng cho paired bootstrap ở bước so sánh.
  records = []
 
  for index, (reference, hypothesis) in enumerate(zip(references, hypotheses)):
    records.append({
        'qa_id': qa_ids[index] if qa_ids is not None else index,
        'rouge_l': rouge_l(reference, hypothesis, **tokenize_kwargs),
        'token_f1': token_f1(reference, hypothesis, **tokenize_kwargs),
        # BLEU mức câu chỉ để bootstrap; con số báo cáo lấy từ corpus_bleu.
        'sentence_bleu1': corpus_bleu([reference], [hypothesis], max_n=1, **tokenize_kwargs),
        'sentence_bleu4': corpus_bleu([reference], [hypothesis], max_n=4, **tokenize_kwargs),
    })
 
  per_example = pd.DataFrame(records)
 
  if use_bertscore:
    per_example['bertscore'] = compute_bertscore(references, hypotheses, model_type = bertscore_model)
 
  return per_example


def summarize_generations(references: list[str], hypotheses: list[str], per_example: pd.DataFrame = None, **tokenize_kwargs) -> dict:
  # Con số báo cáo. BLEU tính ở mức corpus, ROUGE-L và BERTScore lấy trung bình.
  summary = {
      'BLEU1': round(corpus_bleu(references, hypotheses, max_n=1, **tokenize_kwargs) * 100, 2),
      'BLEU4': round(corpus_bleu(references, hypotheses, max_n=4, **tokenize_kwargs) * 100, 2),
  }
 
  if per_example is not None:
    summary['ROUGE-L'] = round(per_example['rouge_l'].mean() * 100, 2)
    summary['F1'] = round(per_example['token_f1'].mean() * 100, 2)
 
    if 'bertscore' in per_example:
      summary['BERTScore'] = round(per_example['bertscore'].mean() * 100, 2)
 
  return summary


# 4. Baseline tầm thường — sàn để biết model có học được gì không
def trivial_baselines(frame: pd.DataFrame) -> dict:
  return {
      'copy_question': frame['source_question'].tolist(),
      'copy_context': frame['source_context'].tolist(),
  }
 
 
# 5. Kiểm định thống kê
def paired_bootstrap(per_example_a: pd.DataFrame, per_example_b: pd.DataFrame, metric: str = 'rouge_l', number_of_resamples: int = 10000, random_seed: int = 42) -> dict:
  merged = per_example_a[['qa_id', metric]].merge(per_example_b[['qa_id', metric]], on='qa_id', suffixes=('_a', '_b'))
 
  if merged.empty:
    raise ValueError("Hai bảng không có qa_id chung.")
 
  differences = (merged[f'{metric}_a'] - merged[f'{metric}_b']).to_numpy()
 
  observed_difference = float(differences.mean())
 
  rng = np.random.default_rng(random_seed)
 
  indices = rng.integers(0, len(differences), size=(number_of_resamples, len(differences)))
 
  resampled_means = differences[indices].mean(axis=1)
 
  centered = resampled_means - observed_difference
 
  p_value = float((np.abs(centered) >= abs(observed_difference)).mean())
 
  return {
      'metric': metric,
      'number_of_examples': len(merged),
      'mean_a': float(merged[f'{metric}_a'].mean()),
      'mean_b': float(merged[f'{metric}_b'].mean()),
      'difference': observed_difference,
      'ci_lower_95': float(np.percentile(resampled_means, 2.5)),
      'ci_upper_95': float(np.percentile(resampled_means, 97.5)),
      'p_value': p_value,
      'significant_at_0.05': bool(p_value < 0.05),
  }
 
# 6. Lưu kết quả
def save_run(output_dir: Path, run_name: str, predictions_by_split: dict, per_example_by_split: dict, summary_by_split: dict, config: dict) -> pd.DataFrame:
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
 
  summary_records = []
 
  for split_name, summary in summary_by_split.items():
    summary_records.append({'run_name': run_name, 'split': split_name, **summary})
 
  for split_name, per_example in per_example_by_split.items():
    per_example.to_parquet(output_dir / f'{run_name}__{split_name}_per_example.parquet', index=False)
 
  for split_name, predictions in predictions_by_split.items():
    predictions.to_parquet(output_dir / f'{run_name}__{split_name}_predictions.parquet', index=False)
 
  summary_dataframe = pd.DataFrame(summary_records)
 
  summary_dataframe.to_csv(output_dir / f'{run_name}__summary.csv', index=False, encoding='utf-8-sig')
 
  with open(output_dir / f'{run_name}__config.json', 'w', encoding='utf-8') as file:
    json.dump(config, file, ensure_ascii=False, indent=2, default=str)
 
  return summary_dataframe
 
 
def collect_leaderboard(output_dir: Path) -> pd.DataFrame:
  key_columns = ['run_name', 'split', 'BLEU1', 'BLEU4', 'ROUGE-L', 'BERTScore', 'F1']
 
  summary_paths = sorted(Path(output_dir).glob('*__summary.csv'))
 
  if not summary_paths:
    return pd.DataFrame(columns=key_columns)
 
  leaderboard = pd.concat([pd.read_csv(path) for path in summary_paths], ignore_index=True)
 
  available = [column for column in key_columns if column in leaderboard.columns]
 
  return (leaderboard[available].sort_values(['split', 'BLEU1'], ascending=[True, False]).reset_index(drop=True))