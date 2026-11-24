# Input = ['query_id', 'doc_id', 'rank'] : DataFrame 


from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

MODULE_VERSION = 3

TOP_K_LIST = [1, 3, 5, 10, 15, 20, 25, 30]
RANKING_DEPTH = 50



def require_version(minimum_version: int) -> None:
  if MODULE_VERSION < minimum_version:
    raise ImportError(
        f"retriever_common phiên bản {MODULE_VERSION}, cần >= {minimum_version}.\n"
        f"Nếu đã ghi đè file trên Drive, module cũ vẫn còn trong bộ nhớ. Chạy:\n"
        f"    import importlib, retriever_common\n"
        f"    importlib.reload(retriever_common)\n"
        f"    rc = retriever_common\n"
        f"Hoặc Runtime -> Restart session rồi chạy lại."
    )

  print(f"retriever_common v{MODULE_VERSION} (cần >= {minimum_version})")



# 1. Metrics
def build_qrels_map(qrels_dataframe: pd.DataFrame) -> dict:
  qrels_map = defaultdict(set)

  for query_id, doc_id, relevance in zip(
      qrels_dataframe['query_id'], qrels_dataframe['doc_id'], qrels_dataframe['relevance']
  ):
    if relevance > 0:
      qrels_map[query_id].add(doc_id)

  return dict(qrels_map)


def recall_at_k(gold_relevant_doc_ids: set, retrieved_doc_ids: list, k: int) -> float:
  if not gold_relevant_doc_ids:
    return 0.0

  return len(set(retrieved_doc_ids[:k]) & gold_relevant_doc_ids) / len(gold_relevant_doc_ids)


def hit_rate_at_k(gold_relevant_doc_ids: set, retrieved_doc_ids: list, k: int) -> float:
  return float(bool(set(retrieved_doc_ids[:k]) & gold_relevant_doc_ids))

def precision_at_k(gold_relevant_doc_ids: set, retrieved_doc_ids: list, k: int) -> float:
  if k <= 0:
    return 0.0
 
  return len(set(retrieved_doc_ids[:k]) & gold_relevant_doc_ids) / k

def p_at_k(gold_relevant_doc_ids: set, retrieved_doc_ids: list, k: int, n: int) -> float:
  # n: number of queries in data set 
  if n <= 0:
    return 0.0

  p = 1 if len(set(retrieved_doc_ids[:k]) & gold_relevant_doc_ids) > 0 else 0

  return p 

def reciprocal_rank_at_k(gold_relevant_doc_ids: set, retrieved_doc_ids: list, k: int) -> float:
  for rank, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
    if doc_id in gold_relevant_doc_ids:
      return 1.0 / rank

  return 0.0


def ndcg_at_k(gold_relevant_doc_ids: set, retrieved_doc_ids: list, k: int) -> float:
  if not gold_relevant_doc_ids:
    return 0.0

  actual_dcg = sum(
      1.0 / np.log2(rank + 1)
      for rank, doc_id in enumerate(retrieved_doc_ids[:k], start=1)
      if doc_id in gold_relevant_doc_ids
  )

  ideal_dcg = sum(
      1.0 / np.log2(rank + 1)
      for rank in range(1, min(len(gold_relevant_doc_ids), k) + 1)
  )

  return actual_dcg / ideal_dcg if ideal_dcg else 0.0


def evaluation(ranking_dataframe: pd.DataFrame, qrels_map: dict,
               top_k_list: list[int] = None) -> pd.DataFrame:
  top_k_list = top_k_list or TOP_K_LIST

  num_of_query = ranking_dataframe['query_id'].nunique()

  ranking_dataframe = ranking_dataframe.sort_values(['query_id', 'rank'])

  ranked_doc_ids_by_query = (
      ranking_dataframe.groupby('query_id')['doc_id'].apply(list).to_dict()
  )

  records = []

  for query_id, retrieved_doc_ids in ranked_doc_ids_by_query.items():
    gold_relevant_doc_ids = qrels_map.get(query_id)

    if not gold_relevant_doc_ids:
      continue

    record = {'query_id': query_id, 'number_of_positives': len(gold_relevant_doc_ids)}

    for k in top_k_list:
      record[f'recall@{k}'] = recall_at_k(gold_relevant_doc_ids, retrieved_doc_ids, k)
      record[f'p@{k}'] = p_at_k(gold_relevant_doc_ids, retrieved_doc_ids, k, num_of_query)
      record[f'hit@{k}'] = hit_rate_at_k(gold_relevant_doc_ids, retrieved_doc_ids, k)
      record[f'precision@{k}'] = precision_at_k(gold_relevant_doc_ids, retrieved_doc_ids, k)
      record[f'mrr@{k}'] = reciprocal_rank_at_k(gold_relevant_doc_ids, retrieved_doc_ids, k)
      record[f'ndcg@{k}'] = ndcg_at_k(gold_relevant_doc_ids, retrieved_doc_ids, k)

    records.append(record)

  return pd.DataFrame(records)

def summarize(per_query_dataframe: pd.DataFrame) -> pd.Series:
  return per_query_dataframe.drop(columns=['query_id', 'number_of_positives']).mean()


def assert_metrics_are_sane(per_query_dataframe: pd.DataFrame, top_k_list: list[int] = None) -> None:
  """Bắt lại đúng lỗi đã có ở bản BM25 cũ: MRR không cắt theo k."""
  top_k_list = top_k_list or TOP_K_LIST

  single_positive = per_query_dataframe[per_query_dataframe['number_of_positives'] == 1]

  assert np.allclose(single_positive['mrr@1'], single_positive['recall@1']), (
      "MRR@1 khác Recall@1 — hàm reciprocal_rank_at_k có lỗi."
  )

  mean_mrr = [per_query_dataframe[f'mrr@{k}'].mean() for k in top_k_list]

  assert mean_mrr == sorted(mean_mrr), "MRR không tăng theo k."
  assert len(set(np.round(mean_mrr, 6))) > 1, "MRR không đổi theo k — vẫn còn lỗi cũ."



# 2. Chunking — cho model có max_seq_length ngắn (vietnamese-bi-encoder: 256)

def chunk_text_by_tokens(text: str, tokenizer, max_tokens: int, overlap_tokens: int) -> list[str]:
  """Cắt theo token của CHÍNH tokenizer của model, không phải theo ký tự hay theo từ."""
  token_ids = tokenizer.encode(text, add_special_tokens=False)

  # Trừ hao chỗ cho các special token mà tokenizer sẽ thêm vào lúc encode thật.
  effective_max_tokens = max(1, max_tokens - 4)

  if len(token_ids) <= effective_max_tokens:
    return [text]

  stride = max(1, effective_max_tokens - overlap_tokens)

  chunks = []

  for start_index in range(0, len(token_ids), stride):
    window_ids = token_ids[start_index:start_index + effective_max_tokens]

    if not window_ids:
      break

    chunks.append(tokenizer.decode(window_ids, skip_special_tokens=True))

    if start_index + effective_max_tokens >= len(token_ids):
      break

  return chunks


def build_chunk_table(corpus: pd.DataFrame, tokenizer, max_tokens: int, overlap_tokens: int,
                      text_column: str = 'embedding_text') -> pd.DataFrame:
  chunk_records = []

  for doc_id, text in zip(corpus['doc_id'], corpus[text_column]):
    for chunk_index, chunk_text in enumerate(
        chunk_text_by_tokens(text, tokenizer, max_tokens, overlap_tokens)
    ):
      chunk_records.append({
          'chunk_id': f"{doc_id}__c{chunk_index}",
          'doc_id': doc_id,
          'chunk_index': chunk_index,
          'chunk_text': chunk_text,
      })

  return pd.DataFrame(chunk_records)


# 3. Từ ma trận điểm -> DataFrame ranking

def rankings_from_score_matrix(score_matrix: np.ndarray, query_ids: list, doc_ids: np.ndarray,
                               split_name: str, depth: int = RANKING_DEPTH) -> pd.DataFrame:
  # score_matrix: (số câu hỏi, số passage). Trả về DataFrame ranking chuẩn.
  depth = min(depth, score_matrix.shape[1])

  # argpartition lấy top-k không cần sắp xếp toàn bộ, rồi mới sort trong k phần tử đó.
  top_indices = np.argpartition(-score_matrix, depth - 1, axis=1)[:, :depth]

  row_indices = np.arange(score_matrix.shape[0])[:, None]
  top_scores = score_matrix[row_indices, top_indices]

  # kind='stable' để các passage bằng điểm luôn giữ thứ tự corpus.
  order_within_top = np.argsort(-top_scores, axis=1, kind='stable')

  sorted_indices = top_indices[row_indices, order_within_top]
  sorted_scores = top_scores[row_indices, order_within_top]

  number_of_queries = score_matrix.shape[0]

  return pd.DataFrame({
      'query_id': np.repeat(np.asarray(query_ids), depth),
      'doc_id': doc_ids[sorted_indices].reshape(-1),
      'rank': np.tile(np.arange(1, depth + 1), number_of_queries),
      'score': sorted_scores.reshape(-1).astype(float),
      'split': split_name,
  })


def max_pool_chunk_scores(chunk_score_matrix: np.ndarray, chunk_table: pd.DataFrame,
                          doc_ids: np.ndarray) -> np.ndarray:
  # Gộp điểm chunk về điểm document: score(doc) = max(score(chunk thuộc doc)).
  doc_id_to_column = {doc_id: column for column, doc_id in enumerate(doc_ids)}

  chunk_columns = chunk_table['doc_id'].map(doc_id_to_column).to_numpy()

  document_score_matrix = np.full(
      (chunk_score_matrix.shape[0], len(doc_ids)), -np.inf, dtype=chunk_score_matrix.dtype
  )

  np.maximum.at(document_score_matrix.T, chunk_columns, chunk_score_matrix.T)

  return document_score_matrix



# Đánh giá Bi-Encoder 

def encode_corpus_matrix(model, corpus: pd.DataFrame, prepare_passage, *,
                         use_chunking: bool = False, chunk_size: int = None,
                         chunk_overlap: int = 64, max_passage_length: int = 512,
                         batch_size: int = 16, normalize_embeddings: bool = True):
  """Trả về (embedding_matrix, chunk_table). chunk_table = None khi không chunk."""
  if use_chunking:
    chunk_table = build_chunk_table(
        corpus.assign(prepared_text=corpus['embedding_text'].map(prepare_passage)),
        tokenizer=model.tokenizer,
        max_tokens=chunk_size,
        overlap_tokens=chunk_overlap,
        text_column='prepared_text',
    )
    texts_to_encode = chunk_table['chunk_text'].tolist()
    model.max_seq_length = chunk_size
  else:
    chunk_table = None
    texts_to_encode = [prepare_passage(text) for text in corpus['embedding_text']]
    model.max_seq_length = max_passage_length
 
  embedding_matrix = model.encode(
      texts_to_encode, batch_size=batch_size, normalize_embeddings=normalize_embeddings,
      convert_to_numpy=True, show_progress_bar=False,
  ).astype(np.float32)
 
  return embedding_matrix, chunk_table
  
 
 
def evaluate_bi_encoder(model, corpus: pd.DataFrame, queries_dataframe: pd.DataFrame,
                        qrels_dataframe: pd.DataFrame, prepare_query, prepare_passage, *,
                        split_name: str, use_chunking: bool = False, chunk_size: int = None,
                        chunk_overlap: int = 64, max_passage_length: int = 512,
                        max_query_length: int = 64, batch_size_passage: int = 16,
                        batch_size_query: int = 64, query_prompt_name=None,
                        normalize_embeddings: bool = True,
                        passage_embeddings=None, chunk_table=None):
  
  doc_id_array = corpus['doc_id'].to_numpy()
 
  if passage_embeddings is None:
    passage_embeddings, chunk_table = encode_corpus_matrix(
        model, corpus, prepare_passage,
        use_chunking=use_chunking, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        max_passage_length=max_passage_length, batch_size=batch_size_passage,
        normalize_embeddings=normalize_embeddings,
    )
 
  model.max_seq_length = max_query_length
 
  encode_kwargs = dict(batch_size=batch_size_query, normalize_embeddings=normalize_embeddings,
                       convert_to_numpy=True, show_progress_bar=False)
 
  if query_prompt_name is not None:
    encode_kwargs['prompt_name'] = query_prompt_name
 
  query_embeddings = model.encode(
      [prepare_query(text) for text in queries_dataframe['query']], **encode_kwargs
  ).astype(np.float32)
 
  score_matrix = query_embeddings @ passage_embeddings.T
 
  if use_chunking:
    score_matrix = max_pool_chunk_scores(score_matrix, chunk_table, doc_id_array)
 
  ranking_dataframe = rankings_from_score_matrix(
      score_matrix, queries_dataframe['query_id'].tolist(), doc_id_array, split_name
  )
 
  per_query_dataframe = evaluation(ranking_dataframe, build_qrels_map(qrels_dataframe))
 
  return ranking_dataframe, per_query_dataframe, passage_embeddings, chunk_table



# Kiểm định ý nghĩa thống kê 
def paired_bootstrap(per_query_a: pd.DataFrame, per_query_b: pd.DataFrame,
                     metric: str = 'recall@1', number_of_resamples: int = 10000,
                     random_seed: int = 42) -> dict:
  """So sánh hai hệ thống trên CÙNG tập câu hỏi.
 
  Trả về chênh lệch trung bình, khoảng tin cậy 95% và p-value hai phía.
  Nếu khoảng tin cậy chứa 0 thì chênh lệch quan sát được không phân biệt được với nhiễu.
  """
  merged = per_query_a[['query_id', metric]].merge(
      per_query_b[['query_id', metric]], on='query_id', suffixes=('_a', '_b')
  )
 
  if merged.empty:
    raise ValueError("Hai bảng không có query_id chung — kiểm tra lại split.")
 
  differences = (merged[f'{metric}_a'] - merged[f'{metric}_b']).to_numpy()
 
  observed_difference = float(differences.mean())
 
  rng = np.random.default_rng(random_seed)
 
  indices = rng.integers(0, len(differences), size=(number_of_resamples, len(differences)))
 
  resampled_means = differences[indices].mean(axis=1)
 
  # p-value: dịch phân phối về 0 rồi đếm phần đuôi vượt quá chênh lệch quan sát được.
  centered = resampled_means - observed_difference
 
  p_value = float((np.abs(centered) >= abs(observed_difference)).mean())
 
  return {
      'metric': metric,
      'number_of_queries': len(merged),
      'mean_a': float(merged[f'{metric}_a'].mean()),
      'mean_b': float(merged[f'{metric}_b'].mean()),
      'difference': observed_difference,
      'ci_lower_95': float(np.percentile(resampled_means, 2.5)),
      'ci_upper_95': float(np.percentile(resampled_means, 97.5)),
      'p_value': p_value,
      'significant_at_0.05': bool(p_value < 0.05),
  }


# 4. Đo tốc độ — cùng cách với BM25 để các con số so sánh được

def measure_latency(encode_query_function, search_function, query_texts: list[str],
                    warmup: int = 10) -> dict:
  # encode_query_function: str -> vector. search_function: vector -> ranking (top-k).
  for query_text in query_texts[:warmup]:
    search_function(encode_query_function(query_text))

  encode_elapsed = 0.0
  search_elapsed = 0.0

  for query_text in query_texts:
    start_time = time.perf_counter()
    query_vector = encode_query_function(query_text)
    encode_elapsed += time.perf_counter() - start_time

    start_time = time.perf_counter()
    search_function(query_vector)
    search_elapsed += time.perf_counter() - start_time

  number_of_queries = len(query_texts)
  total_elapsed = encode_elapsed + search_elapsed

  return {
      'number_of_queries': number_of_queries,
      'encode_ms_per_query': round(encode_elapsed / number_of_queries * 1000, 3),
      'search_ms_per_query': round(search_elapsed / number_of_queries * 1000, 3),
      'latency_ms_per_query': round(total_elapsed / number_of_queries * 1000, 3),
      'queries_per_second': round(number_of_queries / total_elapsed, 1),
  }



# 5. Lưu kết quả theo cùng một bố cục cho mọi retriever

def save_run(output_dir: Path, run_name: str, ranking_by_split: dict,
             per_query_by_split: dict, config: dict) -> pd.DataFrame:
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  summary_records = []

  for split_name, per_query_dataframe in per_query_by_split.items():
    summary_records.append({
        'run_name': run_name,
        'split': split_name,
        'number_of_queries': len(per_query_dataframe),
        **summarize(per_query_dataframe).to_dict(),
    })

    per_query_dataframe.to_parquet(
        output_dir / f'{run_name}__{split_name}_metrics_per_query.parquet', index=False
    )

  for split_name, ranking_dataframe in ranking_by_split.items():
    ranking_dataframe.to_parquet(
        output_dir / f'{run_name}__{split_name}_ranking.parquet', index=False
    )

  summary_dataframe = pd.DataFrame(summary_records)

  summary_dataframe.to_csv(output_dir / f'{run_name}__summary.csv', index=False, encoding='utf-8-sig')

  with open(output_dir / f'{run_name}__config.json', 'w', encoding='utf-8') as file:
    json.dump(config, file, ensure_ascii=False, indent=2, default=str)

  return summary_dataframe


def collect_leaderboard(output_dir: Path) -> pd.DataFrame:
  key_columns = ['run_name', 'split', 'number_of_queries', 'p@1', 'p@5', 'p@10', 'p@15', 'p@20', 
                 'recall@1', 'recall@5', 'recall@10', 'mrr@1', 'mrr@10', 'ndcg@1','ndcg@10']

  summary_paths = sorted(Path(output_dir).glob('*__summary.csv'))

  if not summary_paths:
    return pd.DataFrame(columns=key_columns)

  leaderboard = pd.concat(
      [pd.read_csv(path) for path in summary_paths], ignore_index=True
  )

  available_columns = [column for column in key_columns if column in leaderboard.columns]

  return (leaderboard[available_columns]
          .sort_values(['split', 'p@1'], ascending=[True, False])
          .reset_index(drop=True))