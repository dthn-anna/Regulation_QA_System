from __future__ import annotations

from pathlib import Path
import hashlib
import json
import re
import unicodedata

import numpy as np
import pandas as pd

PIPELINE_VERSION = 1

RANKING_COLUMNS = ['query_id', 'doc_id', 'rank', 'score', 'split']

RANKING_DEPTH = 50

MISSING_RANK = 10 ** 6


def require_version(minimum_version: int) -> None:
  if PIPELINE_VERSION < minimum_version:
    raise ImportError(
        f"rag_pipeline phiên bản {PIPELINE_VERSION}, cần >= {minimum_version}.\n"
        f"\timport importlib, rag_pipeline\n"
        f"\timportlib.reload(rag_pipeline)\n"
    )

  print(f"rag_pipeline v{PIPELINE_VERSION} (cần >= {minimum_version})")



def normalize_text(text: str) -> str:
  if text is None or (isinstance(text, float) and np.isnan(text)):
    return ""

  text = unicodedata.normalize("NFKC", str(text))
  text = text.replace("\u00a0", " ")
  text = text.replace("\r\n", "\n").replace("\r", "\n")
  text = re.sub(r"[ \t]+", " ", text)
  text = re.sub(r" *\n *", "\n", text)
  text = re.sub(r"\n{3,}", "\n\n", text)

  return text.strip()


def normalize_for_comparison(text: str) -> str:
  text = normalize_text(text).lower()

  return re.sub(r"\s+", " ", text).strip()


def stable_id(prefix: str, *values: str, length: int = 20) -> str:
  payload = "|||".join(normalize_for_comparison(value) for value in values)
  digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]

  return f"{prefix}_{digest}"


def query_id_of(question: str) -> str:
  """ID của câu hỏi người dùng gõ vào demo, sinh đúng cách như lúc tiền xử lý.

  Nhờ vậy nếu câu hỏi trùng một câu trong tập test thì so được ngay kết quả.
  """
  return stable_id("QUERY", question)


# 2. Corpus

class CorpusStore:
  """Bọc corpus.parquet."""

  def __init__(self, corpus: pd.DataFrame):
    required = {'doc_id', 'document', 'article', 'context', 'embedding_text', 'title'}
    missing = required - set(corpus.columns)

    if missing:
      raise ValueError(f"corpus thiếu cột: {sorted(missing)}")

    if not corpus['doc_id'].is_unique:
      raise ValueError("corpus có doc_id trùng.")

    self.frame = corpus.reset_index(drop=True)
    self.doc_ids = self.frame['doc_id'].to_numpy()

    self._columns = {
        name: dict(zip(self.frame['doc_id'], self.frame[name].astype(str)))
        for name in ['context', 'document', 'article', 'title', 'embedding_text']
        if name in self.frame.columns
    }

    self._title = self._columns.get('title', {})
    self._document = self._columns.get('document', {})

  def __len__(self) -> int:
    return len(self.frame)

  def title(self, doc_id: str) -> str:
    return self._title.get(doc_id, '')

  def embedding_text(self, doc_id: str) -> str:
    """Chuỗi mà retriever đánh chỉ mục. KHÔNG đưa chuỗi này cho generator."""
    return self._embedding_text.get(doc_id, '')

  def passage_for_generator(self, doc_id: str, passage_column: str) -> str:
    """Chuỗi mà generator nhìn thấy — đọc THẲNG một cột của corpus.

    `passage_column` lấy từ prompt_config.json của checkpoint, không hardcode.
    Không nối chuỗi ở đây: mỗi công thức nối là một đường có thể trôi khỏi
    công thức dùng lúc fine-tune.
    """
    column = self._columns.get(passage_column)

    if column is None:
      raise ValueError(f"corpus không có cột '{passage_column}'. Có: {sorted(self._columns)}")

    return column.get(doc_id, '')

  def build_generator_context(self, doc_ids, passage_column: str, separator: str = '\n\n') -> str:
    return separator.join(self.passage_for_generator(doc_id, passage_column) for doc_id in doc_ids)



# 3. Retriever

def tokenizer_vietnamese(text: str) -> list[str]:
  """Chép nguyên từ BM25.ipynb. Đổi hàm này là đổi toàn bộ điểm BM25."""
  from pyvi import ViTokenizer

  text = str(text).strip().lower()

  if not text:
    return []

  text = re.sub(r"[^\w\s]", " ", text)

  return ViTokenizer.tokenize(text).split()


class BM25Retriever:
  def __init__(self, corpus_store: CorpusStore, k1: float, b: float, text_column: str):
    from rank_bm25 import BM25Okapi

    self.corpus_store = corpus_store
    self.config = {'k1': k1, 'b': b, 'text_column': text_column}

    texts = corpus_store.frame[text_column].tolist()
    self.index = BM25Okapi([tokenizer_vietnamese(text) for text in texts], k1=k1, b=b)

  @classmethod
  def from_config(cls, corpus_store: CorpusStore, config: dict) -> "BM25Retriever":
    return cls(corpus_store, k1=config['k1'], b=config['b'], text_column=config['text_column'])

  def score_all(self, query_text: str) -> np.ndarray:
    return np.asarray(self.index.get_scores(tokenizer_vietnamese(query_text)))

  def search(self, query_text: str, depth: int = RANKING_DEPTH) -> pd.DataFrame:
    scores = self.score_all(query_text)

    # kind='stable' để doc bằng điểm giữ thứ tự corpus -> tái lập được.
    top_indices = np.argsort(-scores, kind='stable')[:depth]

    return pd.DataFrame({
        'doc_id': self.corpus_store.doc_ids[top_indices],
        'rank': np.arange(1, len(top_indices) + 1),
        'score': scores[top_indices].astype(float),
    })


class DenseRetriever:

  def __init__(self, corpus_store: CorpusStore, model, config: dict):
    self.corpus_store = corpus_store
    self.model = model
    self.config = config

    self.passage_embeddings = None
    self.chunk_table = None

  @classmethod
  def from_config(cls, corpus_store: CorpusStore, config: dict, model_dir: str = None) -> "DenseRetriever":
    from sentence_transformers import SentenceTransformer

    model_path = str(model_dir or config['model_save_dir'])
    model = SentenceTransformer(model_path, trust_remote_code=True)

    return cls(corpus_store, model, config)

  def prepare_query(self, text: str) -> str:
    text = str(text)

    if self.config.get('use_word_segmentation'):
      from pyvi import ViTokenizer
      text = ViTokenizer.tokenize(text)

    return self.config.get('query_prefix', '') + text

  def prepare_passage(self, text: str) -> str:
    text = str(text)

    if self.config.get('use_word_segmentation'):
      from pyvi import ViTokenizer
      text = ViTokenizer.tokenize(text)

    return self.config.get('passage_prefix', '') + text

  def _cache_signature(self) -> str:
    """Vân tay của mọi thứ ảnh hưởng tới vector. Đổi một tham số là cache hết hiệu lực.

    Không có cái này thì sau khi đổi checkpoint hoặc đổi chunk_size, hệ thống sẽ
    lặng lẽ phục vụ vector cũ — sai mà không báo lỗi.
    """
    relevant = {key: self.config.get(key) for key in [
        'run_name', 'model_save_dir', 'passage_prefix', 'use_word_segmentation',
        'normalize_embeddings', 'use_chunking', 'chunk_size', 'chunk_overlap',
        'max_passage_length',
    ]}

    relevant['number_of_passages'] = len(self.corpus_store)
    relevant['corpus_fingerprint'] = hashlib.sha256(
        "|".join(self.corpus_store.doc_ids).encode('utf-8')
    ).hexdigest()[:16]

    payload = json.dumps(relevant, sort_keys=True, ensure_ascii=False)

    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:20]

  def build_index(self, batch_size: int = 16, cache_dir=None) -> None:
    """Mã hoá corpus. Trên GPU vài giây; trên CPU với bge-m3 có thể vài phút, nên khi chạy local nên truyền cache_dir để chỉ mã hoá đúng một lần.
    """
    import retriever_common as rc

    cache_path = None

    if cache_dir is not None:
      cache_dir = Path(cache_dir)
      cache_dir.mkdir(parents=True, exist_ok=True)

      cache_path = cache_dir / f'passage_embeddings__{self._cache_signature()}.npz'

      if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)

        self.passage_embeddings = cached['embeddings']
        self.chunk_table = pd.DataFrame(cached['chunk_table'], columns=list(cached['chunk_columns']))

        print(f"Nạp vector từ cache: {cache_path.name}")

        return

    self.passage_embeddings, self.chunk_table = rc.encode_corpus_matrix(
        self.model, self.corpus_store.frame, self.prepare_passage,
        use_chunking=self.config.get('use_chunking', False),
        chunk_size=self.config.get('chunk_size'),
        chunk_overlap=self.config.get('chunk_overlap', 64),
        max_passage_length=self.config.get('max_passage_length', 512),
        batch_size=batch_size,
        normalize_embeddings=self.config.get('normalize_embeddings', True),
    )

    if cache_path is not None:
      np.savez_compressed(cache_path, embeddings=self.passage_embeddings, chunk_table=self.chunk_table.to_numpy(), chunk_columns=np.array(list(self.chunk_table.columns)))

      print(f"Đã lưu cache vector: {cache_path.name}")

  def score_all(self, query_text: str) -> np.ndarray:
    import retriever_common as rc

    if self.passage_embeddings is None:
      self.build_index()

    self.model.max_seq_length = self.config.get('max_query_length', 64)

    encode_kwargs = dict(normalize_embeddings=self.config.get('normalize_embeddings', True), convert_to_numpy=True, show_progress_bar=False)

    if self.config.get('query_prompt_name'):
      encode_kwargs['prompt_name'] = self.config['query_prompt_name']

    query_vector = self.model.encode([self.prepare_query(query_text)], **encode_kwargs).astype(np.float32)

    score_matrix = query_vector @ self.passage_embeddings.T

    if self.config.get('use_chunking'):
      score_matrix = rc.max_pool_chunk_scores(score_matrix, self.chunk_table, self.corpus_store.doc_ids)

    return score_matrix[0]

  def search(self, query_text: str, depth: int = RANKING_DEPTH) -> pd.DataFrame:
    scores = self.score_all(query_text)
    top_indices = np.argsort(-scores, kind='stable')[:depth]

    return pd.DataFrame({
        'doc_id': self.corpus_store.doc_ids[top_indices],
        'rank': np.arange(1, len(top_indices) + 1),
        'score': scores[top_indices].astype(float),
    })


def build_candidate_table(bm25_ranking: pd.DataFrame, dense_ranking: pd.DataFrame) -> pd.DataFrame:
  merged = pd.merge(
      bm25_ranking.rename(columns={'rank': 'rank_bm25', 'score': 'score_bm25'}),
      dense_ranking.rename(columns={'rank': 'rank_dense', 'score': 'score_dense'}),
      on=['query_id', 'doc_id'], how='outer',
  )

  merged['rank_bm25'] = merged['rank_bm25'].fillna(MISSING_RANK).astype(int)
  merged['rank_dense'] = merged['rank_dense'].fillna(MISSING_RANK).astype(int)

  for column in ['score_bm25', 'score_dense']:
    per_query_min = merged.groupby('query_id')[column].transform('min')
    merged[column] = merged[column].fillna(per_query_min)

  for column in ['score_bm25', 'score_dense']:
    grouped = merged.groupby('query_id')[column]

    minimum = grouped.transform('min')
    maximum = grouped.transform('max')
    spread = (maximum - minimum).replace(0, np.nan)

    merged[f'{column}_minmax'] = ((merged[column] - minimum) / spread).fillna(0.5)

    mean = grouped.transform('mean')
    std = grouped.transform('std').replace(0, np.nan)

    merged[f'{column}_z'] = ((merged[column] - mean) / std).fillna(0.0)

  return merged


def fuse(table: pd.DataFrame, method: str, **parameters) -> pd.Series:
  if method == 'rrf':
    k = parameters['k']
    weight = parameters['weight']

    return weight / (k + table['rank_bm25']) + (1.0 - weight) / (k + table['rank_dense'])

  if method == 'weighted_minmax':
    alpha = parameters['alpha']

    return alpha * table['score_bm25_minmax'] + (1.0 - alpha) * table['score_dense_minmax']

  if method == 'weighted_zscore':
    alpha = parameters['alpha']

    return alpha * table['score_bm25_z'] + (1.0 - alpha) * table['score_dense_z']

  if method == 'multiplication':
    return (table['score_bm25_minmax'] + 1e-6) * (table['score_dense_minmax'] + 1e-6)

  if method == 'paper':
    alpha = parameters['alpha']

    return alpha * table['score_bm25'] + (1.0 - alpha) * table['score_dense_minmax']

  raise ValueError(f"Phương pháp không hợp lệ: {method}")


class HybridRetriever:
  def __init__(self, bm25: BM25Retriever, dense: DenseRetriever, method: str, parameters: dict):
    self.bm25 = bm25
    self.dense = dense
    self.method = method
    self.parameters = parameters

  def search(self, query_text: str, depth: int = RANKING_DEPTH) -> pd.DataFrame:
    # Bắt buộc lấy đủ RANKING_DEPTH ở CẢ HAI nhánh trước khi fuse, rồi mới cắt.
    bm25_ranking = self.bm25.search(query_text, RANKING_DEPTH).assign(query_id='_q')
    dense_ranking = self.dense.search(query_text, RANKING_DEPTH).assign(query_id='_q')

    table = build_candidate_table(bm25_ranking, dense_ranking)

    scored = table[['query_id', 'doc_id']].copy()
    scored['score'] = fuse(table, self.method, **self.parameters).to_numpy()

    scored = scored.sort_values(['query_id', 'score'], ascending=[True, False],
                                kind='mergesort')
    scored['rank'] = scored.groupby('query_id').cumcount() + 1

    return (scored[scored['rank'] <= depth].drop(columns=['query_id']).reset_index(drop=True)[['doc_id', 'rank', 'score']])


def load_retriever(serving_config: dict, corpus_store: CorpusStore, dense_model_dir: str = None, cache_dir=None):
  """Dựng lại đúng retriever đã thắng ở bước thực nghiệm."""
  retriever_type = serving_config['retriever_type']

  if retriever_type == 'bm25':
    return BM25Retriever.from_config(corpus_store, serving_config['bm25'])

  dense = DenseRetriever.from_config(corpus_store, serving_config['dense'], dense_model_dir)
  dense.build_index(cache_dir=cache_dir)

  if retriever_type == 'dense':
    return dense

  if retriever_type == 'hybrid':
    bm25 = BM25Retriever.from_config(corpus_store, serving_config['bm25'])

    fusion = dict(serving_config['fusion'])
    method = fusion.pop('method')

    return HybridRetriever(bm25, dense, method, fusion)

  raise ValueError(f"retriever_type không hợp lệ: {retriever_type}")


# 4. Generator
class Generator:
  """Nạp checkpoint kèm prompt_config.json. Prompt KHÔNG được gõ lại bằng tay."""

  def __init__(self, model, tokenizer, prompt_config: dict):
    self.model = model
    self.tokenizer = tokenizer
    self.config = prompt_config

    self.tokenizer.truncation_side = 'left'

    if self.tokenizer.pad_token is None:
      self.tokenizer.pad_token = self.tokenizer.eos_token

  @classmethod
  def load(cls, model_dir: str | Path, device: str = None) -> "Generator":
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_dir = Path(model_dir)

    with open(model_dir / 'prompt_config.json', encoding='utf-8') as file:
      prompt_config = json.load(file)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(str(model_dir), torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32, trust_remote_code=True)

    model = model.to(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model.eval()

    return cls(model, tokenizer, prompt_config)

  def build_user_message(self, question: str, context: str) -> str:
    return self.config['user_template'].format(question=str(question).strip(), context=str(context).strip())

  def render_prompt(self, question: str, context: str) -> str:
    messages = [
        {'role': 'system', 'content': self.config['system_prompt']},
        {'role': 'user', 'content': self.build_user_message(question, context)},
    ]

    template_kwargs = dict(tokenize=False, add_generation_prompt=True)

    if self.config.get('enable_thinking') is not None:
      template_kwargs['enable_thinking'] = self.config['enable_thinking']

    try:
      return self.tokenizer.apply_chat_template(messages, **template_kwargs)
    except Exception:
      # Gemma không nhận vai trò 'system' -> gộp vào lượt user. Giống notebook.
      merged = [{'role': 'user',
                 'content': self.config['system_prompt'] + "\n\n" + self.build_user_message(question, context)}]

      return self.tokenizer.apply_chat_template(merged, **template_kwargs)

  @staticmethod
  def clean_generation(text: str) -> str:
    text = re.sub(r'<think>.*?</think>', ' ', text, flags=re.DOTALL)

    return re.sub(r'\s+', ' ', text).strip()

  def generate(self, prompts: list[str], batch_size: int = 1) -> list[str]:
    import torch

    original_padding_side = self.tokenizer.padding_side
    self.tokenizer.padding_side = 'left'

    outputs = []

    with torch.no_grad():
      for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start:start + batch_size]

        encoded = self.tokenizer(batch_prompts, return_tensors='pt', padding=True, truncation=True, max_length=self.config['max_source_length'], add_special_tokens=False).to(self.model.device)

        generated = self.model.generate(
            **encoded,
            max_new_tokens=self.config['max_target_length'],
            num_beams=self.config.get('num_beams', 1),
            do_sample=self.config.get('do_sample', False),
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        new_tokens = generated[:, encoded['input_ids'].shape[1]:]

        outputs.extend(self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True))

    self.tokenizer.padding_side = original_padding_side

    return [self.clean_generation(text) for text in outputs]


# 5. End-to-end pipeline

class RAGPipeline:
  def __init__(self, corpus_store: CorpusStore, retriever, generator: Generator, serving_config: dict):
    self.corpus_store = corpus_store
    self.retriever = retriever
    self.generator = generator
    self.serving_config = serving_config

  @classmethod
  def load(cls, corpus_path, retriever_serving_path, generator_model_dir, dense_model_dir=None, cache_dir=None, device: str = None) -> "RAGPipeline":
    corpus_store = CorpusStore(pd.read_parquet(corpus_path))

    with open(retriever_serving_path, encoding='utf-8') as file:
      serving_config = json.load(file)

    retriever = load_retriever(serving_config, corpus_store, dense_model_dir, cache_dir=cache_dir)
    generator = Generator.load(generator_model_dir, device=device)

    return cls(corpus_store, retriever, generator, serving_config)

  def retrieve(self, question: str, depth: int = RANKING_DEPTH) -> pd.DataFrame:
    ranking = self.retriever.search(question, depth)
    ranking['title'] = [self.corpus_store.title(doc_id) for doc_id in ranking['doc_id']]

    return ranking

  def answer(self, question: str, top_k: int = 1, display_k: int = 3, abstain_threshold: float = None) -> dict:
    """top_k = số passage đưa vào generator.

    Mặc định 1 vì generator được fine-tune với đúng 1 điều luật; đặt >1 là đưa
    model ra ngoài phân bố huấn luyện và con số sẽ không còn khớp thực nghiệm.
    display_k chỉ để hiển thị trích dẫn trên giao diện, không vào prompt.
    """
    question = normalize_text(question)

    if not question:
      return {'answer': 'Bạn nhập câu hỏi giúp mình nhé.', 'abstained': True,
              'citations': [], 'used_doc_ids': [], 'question': question}

    ranking = self.retrieve(question)

    if abstain_threshold is not None and float(ranking['score'].iloc[0]) < abstain_threshold:
      return {
          'query_id': query_id_of(question), 'question': question,
          'answer': OUT_OF_SCOPE_MESSAGE, 'abstained': True,
          'top_score': float(ranking['score'].iloc[0]),
          'citations': ranking.head(display_k).to_dict('records'),
          'used_doc_ids': [], 'context': '', 'prompt': '',
      }

    selected_doc_ids = ranking['doc_id'].tolist()[:top_k]

    context = self.corpus_store.build_generator_context(selected_doc_ids, passage_column=self.generator.config['passage_column'], separator=self.generator.config.get('context_separator', '\n\n'))

    prompt = self.generator.render_prompt(question, context)
    prediction = self.generator.generate([prompt])[0]

    return {
        'query_id': query_id_of(question),
        'question': question,
        'answer': prediction,
        'abstained': False,
        'top_score': float(ranking['score'].iloc[0]),
        'context': context,
        'used_doc_ids': selected_doc_ids,
        'citations': ranking.head(display_k).to_dict('records'),
        'prompt': prompt,
    }

  def answer_frame(self, frame: pd.DataFrame, top_k: int = 1, batch_size: int = 1) -> pd.DataFrame:
    """Chạy cả một split — dùng để đối chiếu với kết quả thực nghiệm."""
    records = []
    prompts = []

    for question in frame['source_question']:
      ranking = self.retriever.search(question, top_k)
      doc_ids = ranking['doc_id'].tolist()[:top_k]

      context = self.corpus_store.build_generator_context(doc_ids, passage_column=self.generator.config['passage_column'], separator=self.generator.config.get('context_separator', '\n\n'))

      prompts.append(self.generator.render_prompt(question, context))
      records.append({'retrieved_doc_ids': doc_ids, 'source_context': context})

    predictions = self.generator.generate(prompts, batch_size=batch_size)

    result = frame.copy().reset_index(drop=True)
    result = pd.concat([result.drop(columns=['source_context'], errors='ignore'), pd.DataFrame(records)], axis=1)
    result['prediction'] = predictions

    return result


OUT_OF_SCOPE_MESSAGE = (
    "Câu hỏi này nằm ngoài phạm vi các quy định đào tạo mà tôi có. "
    "Bạn thử hỏi lại về quy chế học vụ, đăng ký môn, xét tốt nghiệp, chương trình tài năng hoặc chuẩn ngoại ngữ nhé."
)


def calibrate_abstain_threshold(retriever, in_scope_questions: list[str], out_of_scope_questions: list[str], percentile: float = 5.0) -> dict:
  """Chọn ngưỡng điểm để từ chối, dựa trên hai nhóm câu hỏi đối chứng.

  in_scope_questions   : lấy vài trăm câu từ split test
  out_of_scope_questions: tự viết ~30 câu ngoài miền (thời tiết, nấu ăn, thể thao...)

  Ngưỡng mặc định = phân vị 5 của điểm top-1 trên nhóm in-scope: chấp nhận bỏ sót
  5% câu hợp lệ để chặn phần lớn câu ngoài miền. Điều chỉnh percentile theo khẩu vị.
  """
  def top1_scores(questions):
    return np.array([float(retriever.search(question, 1)['score'].iloc[0]) for question in questions])

  in_scope = top1_scores(in_scope_questions)
  out_of_scope = top1_scores(out_of_scope_questions)

  threshold = float(np.percentile(in_scope, percentile))

  return {
      'threshold': threshold,
      'percentile': percentile,
      'in_scope_mean': float(in_scope.mean()),
      'out_of_scope_mean': float(out_of_scope.mean()),
      'in_scope_bị_từ_chối_nhầm': round(float((in_scope < threshold).mean()) * 100, 2),
      'out_of_scope_bị_chặn_đúng': round(float((out_of_scope < threshold).mean()) * 100, 2),
      'ghi_chú': ('Nếu hai giá trị mean gần nhau thì điểm retriever không phân biệt được trong/ngoài miền — cần thêm bộ phân loại riêng.'),
  }


def verify_ranking_parity(pipeline: RAGPipeline, queries: pd.DataFrame, saved_ranking: pd.DataFrame, top_k: int = 10) -> dict:
  """Chạy lại retriever qua pipeline và so với ranking đã lưu lúc thực nghiệm.

  queries: DataFrame [query_id, query] của split cần kiểm tra.
  saved_ranking: {run}__{split}_ranking.parquet tương ứng.
  """
  saved_top = (saved_ranking.sort_values(['query_id', 'rank']).groupby('query_id')['doc_id'].apply(lambda series: list(series)[:top_k]).to_dict())

  identical = 0
  top1_identical = 0
  mismatches = []

  for query_id, query_text in zip(queries['query_id'], queries['query']):
    recomputed = pipeline.retriever.search(query_text, RANKING_DEPTH)
    recomputed_top = recomputed['doc_id'].tolist()[:top_k]

    expected = saved_top.get(query_id, [])

    if recomputed_top == expected:
      identical += 1
    else:
      mismatches.append({'query_id': query_id,
                         'expected': expected[:3], 'got': recomputed_top[:3]})

    if recomputed_top[:1] == expected[:1]:
      top1_identical += 1

  number_of_queries = len(queries)

  return {
      'number_of_queries': number_of_queries,
      f'top{top_k}_identical_ratio': round(identical / max(number_of_queries, 1), 4),
      'top1_identical_ratio': round(top1_identical / max(number_of_queries, 1), 4),
      'first_mismatches': mismatches[:5],
  }


def verify_context_parity(qa_master: pd.DataFrame, corpus: pd.DataFrame) -> dict:
  """Kiểm tra sai lệch B2: context lúc train (qa_master) vs lúc serve (corpus)."""
  context_by_doc = dict(zip(corpus['doc_id'], corpus['context'].astype(str)))

  differences = [
      doc_id for doc_id, context in zip(qa_master['doc_id'], qa_master['context'].astype(str))
      if context_by_doc.get(doc_id) != context
  ]

  return {
      'number_of_rows': len(qa_master),
      'rows_with_different_context': len(differences),
      'affected_doc_ids': sorted(set(differences))[:10],
      'verdict': ('OK — train và serve dùng cùng chuỗi context.' if not differences else 'LỆCH — phải đổi nguồn context sang corpus và fine-tune lại.'),
  }


def verify_prompt_parity(generator: Generator, sample_row: pd.Series, corpus_store: CorpusStore) -> dict:
  """Dựng prompt theo đường demo và theo context gốc, rồi so byte."""
  gold_context = corpus_store.build_generator_context([sample_row['doc_id']], passage_column=generator.config['passage_column'])

  prompt_from_pipeline = generator.render_prompt(sample_row['source_question'], gold_context)
  prompt_from_training = generator.render_prompt(sample_row['source_question'], sample_row['source_context'])

  return {
      'identical': prompt_from_pipeline == prompt_from_training,
      'length_pipeline': len(prompt_from_pipeline),
      'length_training': len(prompt_from_training),
      'has_think_tag': '<think>' in prompt_from_pipeline,
  }
