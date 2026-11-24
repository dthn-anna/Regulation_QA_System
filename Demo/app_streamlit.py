from pathlib import Path
from threading import Thread
import sys
import json
import html
from datetime import datetime, date 
import uuid 

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

import Demo.settings as settings
import rag_pipeline as rag
 
st.set_page_config(
    page_title="Tư vấn quy định đào tạo UIT",
    page_icon="🎓️",
    layout="centered",
    initial_sidebar_state="expanded",
)
 
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700&family=Lora:ital,wght@0,400;0,500;1,400&display=swap" rel="stylesheet">
 
<style>
  :root {
    --ink:        #101F33;   /* chữ chính, xanh gần đen */
    --ink-soft:   #5A6B80;   /* chú thích */
    --line:       #DFE5EC;   /* đường kẻ, viền */
    --paper:      #F4F6F9;   /* nền trang */
    --surface:    #FFFFFF;
    --accent:     #14509B;   /* xanh học thuật — hỏi của người dùng, dấu nhấn */
    --accent-ink: #0C3A73;
    --seal:       #B0761C;   /* hổ phách — con dấu trên điều luật được dẫn */
    --seal-bg:    #FDF6E7;
 
    --sans: 'Be Vietnam Pro', -apple-system, 'Segoe UI', Roboto, sans-serif;
    --serif: 'Lora', Georgia, 'Times New Roman', serif;
 
    --radius: 14px;
  }
 
  .stApp { background: var(--paper); }
 
  html, body, [class*="css"], .stMarkdown, button, input, textarea {
    font-family: var(--sans);
  }
 
  .block-container { max-width: 780px; padding-top: 1.2rem; padding-bottom: 6rem; }
  header[data-testid="stHeader"] { background: transparent; height: 0; }
  #MainMenu, footer { visibility: hidden; }
 
  /* ---------- Thanh tiêu đề ---------- */
  .app-bar {
    display: flex; align-items: center; gap: 0.7rem;
    padding: 0.15rem 0 0.9rem 0;
    border-bottom: 1px solid var(--line);
    margin-bottom: 1.6rem;
  }
  .app-mark {
    width: 40px; height: 40px; flex: none;
    border-radius: 9px;
    background: var(--accent); color: #fff;
    font-family: var(--serif); font-size: 1.35rem; font-weight: 500;
    display: flex; align-items: center; justify-content: center;
    letter-spacing: 0;
  }
  .app-name { font-size: 1.2rem; font-weight: 600; color: var(--ink); line-height: 1.3; }
  .app-note { font-size: 0.9rem; color: var(--ink-soft); }
 
  /* ---------- Bong bóng ---------- */
  .msg-row { display: flex; margin: 0.75rem 0; align-items: flex-end; gap: 0.55rem; }
  .msg-row.user { justify-content: flex-end; }
  .msg-row.assistant { justify-content: flex-start; }
 
  .avatar {
    width: 28px; height: 28px; flex: none; margin-bottom: 2px;
    border-radius: 50%;
    background: var(--surface); border: 1px solid var(--line); color: var(--accent);
    font-family: var(--serif); font-size: 0.9rem;
    display: flex; align-items: center; justify-content: center;
  }
 
  .bubble {
    max-width: 74%;
    padding: 0.78rem 1.05rem;
    line-height: 1.68;
    font-size: 0.975rem;
    font-weight: 450;
    white-space: pre-wrap;
    word-wrap: break-word;
  }
  .bubble.user {
    background: var(--accent); color: #fff;
    border-radius: var(--radius) var(--radius) 4px var(--radius);
    box-shadow: 0 1px 2px rgba(16,31,51,.16);
  }
  .bubble.assistant {
    background: var(--surface); color: var(--ink);
    border: 1px solid var(--line);
    border-radius: var(--radius) var(--radius) var(--radius) 4px;
    box-shadow: 0 1px 2px rgba(16,31,51,.05);
  }
 
  /* ---------- Thẻ điều luật ---------- */
  .cites { margin: 0.15rem 0 1.5rem 2.35rem; max-width: 74%; }
  .cites-label {
    font-size: 0.72rem; font-weight: 600; letter-spacing: .07em;
    text-transform: uppercase; color: var(--ink-soft);
    margin-bottom: 0.45rem;
  }
 
  details.cite {
    background: var(--surface);
    border: 1px solid var(--line);
    border-left: 3px solid var(--line);
    border-radius: 8px;
    margin-bottom: 0.4rem;
    overflow: hidden;
    transition: border-color .15s ease;
  }
  details.cite[open] { border-left-color: var(--accent); }
  details.cite.used { border-left-color: var(--seal); }
 
  details.cite summary {
    list-style: none; cursor: pointer;
    padding: 0.62rem 0.85rem;
    display: flex; align-items: center; gap: 0.55rem;
    font-size: 0.855rem; color: var(--ink);
  }
  details.cite summary::-webkit-details-marker { display: none; }
  details.cite summary:hover { background: #FAFBFD; }
 
  .cite-title { flex: 1; line-height: 1.45; }
  .cite-doc {
    display: block; font-size: 0.73rem; color: var(--ink-soft);
    margin-top: 1px;
  }
  .cite-seal {
    flex: none; font-size: 0.68rem; font-weight: 600;
    color: var(--seal); background: var(--seal-bg);
    border: 1px solid #EFDFBC; border-radius: 20px;
    padding: 1px 8px; white-space: nowrap;
  }
  .cite-chevron { flex: none; color: var(--ink-soft); font-size: 0.7rem; }
  details.cite[open] .cite-chevron { transform: rotate(90deg); }
 
  .cite-body {
    font-family: var(--serif); font-size: 0.9rem; line-height: 1.78;
    color: #22303F; white-space: pre-wrap;
    padding: 0.2rem 1rem 0.95rem 1rem;
    border-top: 1px solid #F0F3F7;
    margin-top: 0.1rem;
  }
 
  /* ---------- Màn hình trống ---------- */
  .empty { text-align: center; margin: 4.5rem 0 1.8rem; }
  .empty-mark {
    width: 54px; height: 54px; margin: 0 auto 1.1rem;
    border-radius: 15px; background: var(--surface);
    border: 1px solid var(--line);
    font-family: var(--serif); font-size: 1.6rem; color: var(--accent);
    display: flex; align-items: center; justify-content: center;
  }
  .empty h1 { font-size: 1.5rem; font-weight: 600; color: var(--ink); margin: 0 0 0.45rem; }
  .empty p  { font-size: 0.92rem; color: var(--ink-soft); margin: 0 auto; max-width: 27rem; line-height: 1.6; }
  .empty-hint {
    font-size: 0.72rem; font-weight: 600; letter-spacing: .07em;
    text-transform: uppercase; color: var(--ink-soft);
    margin: 2.2rem 0 0.6rem;
  }
 
  /* Nút gợi ý ở màn hình trống */
  .block-container div[data-testid="stButton"] button {
    background: var(--surface); color: var(--ink);
    border: 1px solid var(--line); border-radius: 10px;
    padding: 0.62rem 0.9rem; font-size: 0.885rem; font-weight: 450;
    text-align: left; justify-content: flex-start; line-height: 1.45;
    box-shadow: none; transition: all .15s ease;
    min-height: 76px; align-items: center;
  }
  .block-container div[data-testid="stButton"] button:hover {
    border-color: var(--accent); color: var(--accent-ink);
    transform: translateY(-1px);
    box-shadow: 0 2px 6px rgba(20,80,155,.09);
  }
 
  /* ---------- Ô nhập ---------- */
  div[data-testid="stChatInput"] {
    border: 1px solid var(--line); border-radius: 13px;
    background: var(--surface);
    box-shadow: 0 2px 10px rgba(16,31,51,.06);
  }
  div[data-testid="stChatInput"]:focus-within { border-color: var(--accent); }
  div[data-testid="stChatInput"] textarea { font-size: 0.95rem; }
 
  /* ---------- Sidebar ---------- */
  section[data-testid="stSidebar"] { background: #EDF1F6; border-right: 1px solid var(--line); }
  section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
 
  .side-group {
    font-size: 0.68rem; font-weight: 700; letter-spacing: .09em;
    text-transform: uppercase; color: var(--ink-soft);
    margin: 1.2rem 0 0.3rem 0.35rem;
  }
 
  section[data-testid="stSidebar"] div[data-testid="stButton"] button {
    background: transparent; border: none; box-shadow: none;
    color: var(--ink); font-size: 0.86rem; font-weight: 450;
    text-align: left; justify-content: flex-start;
    padding: 0.42rem 0.6rem; border-radius: 8px; line-height: 1.4;
  }
  section[data-testid="stSidebar"] div[data-testid="stButton"] button:hover {
    background: #E1E8F0; color: var(--ink); transform: none; box-shadow: none;
  }
  section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"] {
    background: var(--surface); color: var(--accent-ink); font-weight: 600;
    box-shadow: 0 1px 2px rgba(16,31,51,.07);
  }
 
  /* Nút tạo cuộc trò chuyện mới — kiểu viền đứt để tách khỏi danh sách */
  .new-chat div[data-testid="stButton"] button {
    border: 1px dashed #B9C6D6; color: var(--accent-ink);
    font-weight: 500; justify-content: center; text-align: center;
    padding: 0.55rem;
  }
 
  @media (prefers-reduced-motion: reduce) {
    * { transition: none !important; animation: none !important; }
  }
</style>
""", unsafe_allow_html=True)
 
 
# Nạp model
@st.cache_resource(show_spinner="Đang nạp mô hình…")
def load_pipeline():
  """cache_resource giữ model trong bộ nhớ qua các lần rerun. Không có nó,
  Streamlit nạp lại model sau mỗi lần gõ."""
  return rag.RAGPipeline.load(
      corpus_path=settings.CORPUS_PATH,
      retriever_serving_path=settings.RETRIEVER_SERVING_PATH,
      generator_model_dir=settings.GENERATOR_MODEL_DIR,
      dense_model_dir=settings.DENSE_MODEL_DIR,
      cache_dir=settings.CACHE_DIR,
      device=settings.DEVICE,
  )
 
 
missing_files = [
    label for label, path in {
        'corpus.parquet': settings.CORPUS_PATH,
        'retriever_serving.json': settings.RETRIEVER_SERVING_PATH,
        'prompt_config.json': settings.GENERATOR_MODEL_DIR / 'prompt_config.json',
    }.items() if not path.exists()
]
 
if missing_files:
  st.error("Thiếu file: " + ", ".join(missing_files))
  st.code(settings.describe())
  st.stop()
 
pipeline = load_pipeline()
 
 
# Lịch sử hội thoại — lưu ra đĩa để không mất khi tắt app
CONVERSATIONS_PATH = settings.CACHE_DIR / 'conversations.json'
 
 
def load_conversations() -> dict:
  if not CONVERSATIONS_PATH.exists():
    return {}
 
  try:
    with open(CONVERSATIONS_PATH, encoding='utf-8') as file:
      return json.load(file)
  except (json.JSONDecodeError, OSError):
    return {}
 
 
def save_conversations() -> None:
  try:
    CONVERSATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
 
    with open(CONVERSATIONS_PATH, 'w', encoding='utf-8') as file:
      json.dump(st.session_state.conversations, file, ensure_ascii=False, indent=2)
  except OSError:
    pass      # không ghi được thì lịch sử chỉ sống trong phiên, không đáng dừng app
 
 
def start_conversation() -> str:
  conversation_id = uuid.uuid4().hex[:12]
 
  st.session_state.conversations[conversation_id] = {
      'title': 'Cuộc trò chuyện mới',
      'created': datetime.now().isoformat(timespec='seconds'),
      'turns': [],
  }
 
  st.session_state.current_id = conversation_id
 
  return conversation_id
 
 
def sorted_conversations():
  return sorted(st.session_state.conversations.items(),
                key=lambda item: item[1].get('created', ''), reverse=True)
 
 
def group_of(created: str) -> str:
  """Nhóm theo ngày để danh sách dài vẫn tra được."""
  try:
    created_date = datetime.fromisoformat(created).date()
  except (ValueError, TypeError):
    return 'Trước đó'
 
  days_ago = (date.today() - created_date).days
 
  if days_ago <= 0:
    return 'Hôm nay'
 
  if days_ago == 1:
    return 'Hôm qua'
 
  if days_ago <= 7:
    return '7 ngày qua'
 
  return 'Trước đó'
 
 
if 'conversations' not in st.session_state:
  st.session_state.conversations = load_conversations()
 
  existing = sorted_conversations()
 
  st.session_state.current_id = existing[0][0] if existing else start_conversation()
 
conversation = st.session_state.conversations[st.session_state.current_id]
 
 
# Sidebar
with st.sidebar:
  st.markdown("<div class='new-chat'>", unsafe_allow_html=True)
 
  if st.button("＋  Cuộc trò chuyện mới", use_container_width=True, key='new_chat'):
    start_conversation()
    save_conversations()
    st.rerun()
 
  st.markdown("</div>", unsafe_allow_html=True)
 
  current_group = None
 
  for conversation_id, record in sorted_conversations():
    group = group_of(record.get('created', ''))
 
    if group != current_group:
      st.markdown(f"<div class='side-group'>{group}</div>", unsafe_allow_html=True)
      current_group = group
 
    is_current = conversation_id == st.session_state.current_id
 
    title = record['title']
    label = title[:32] + "…" if len(title) > 32 else title
 
    label_column, delete_column = st.columns([6, 1], gap='small')
 
    with label_column:
      if st.button(label, key=f'open_{conversation_id}', use_container_width=True,
                   type='primary' if is_current else 'secondary'):
        st.session_state.current_id = conversation_id
        st.rerun()
 
    with delete_column:
      if st.button("✕", key=f'delete_{conversation_id}', help="Xoá cuộc trò chuyện"):
        del st.session_state.conversations[conversation_id]
 
        if is_current:
          remaining = sorted_conversations()
 
          st.session_state.current_id = remaining[0][0] if remaining else start_conversation()
 
        save_conversations()
        st.rerun()
 
 
# Hiển thị
def to_html(text: str) -> str:
  """Model sinh ra văn bản thô — phải escape trước khi nhét vào div, không thì
  một dấu < trong câu trả lời làm vỡ bố cục."""
  return html.escape(str(text)).replace("\n", "<br>")
 
 
BOT_ICON = (
    "<svg viewBox='0 0 24 24' fill='none' stroke-width='1.8' "
    "stroke-linecap='round' stroke-linejoin='round'>"
    "<rect x='4' y='8' width='16' height='12' rx='3'/>"
    "<path d='M12 8V4.5'/><circle cx='12' cy='3.2' r='1.3'/>"
    "<path d='M9.2 13v1.6M14.8 13v1.6'/>"
    "<path d='M1.8 13v3M22.2 13v3'/>"
    "</svg>"
)

def bubble(role: str, text: str) -> str:
  avatar = f"<div class='avatar bot'>{BOT_ICON}</div>" if role == 'assistant' else ""

  return (f"<div class='msg-row {role}'>{avatar}"
          f"<div class='bubble {role}'>{to_html(text)}</div></div>")
 
def split_title(title: str) -> tuple[str, str]:
  """Tiêu đề corpus có dạng '<TÊN VĂN BẢN> - Điều 16. Xử lý học vụ'.
  Tách ra để tên điều đứng trước, tên văn bản làm dòng phụ."""
  if ' - Điều ' in title:
    document_name, article = title.split(' - Điều ', 1)
 
    return 'Điều ' + article, document_name
 
  return title, ''
 
 
def render_citations(citations, used_doc_ids) -> None:
  blocks = ["<div class='cites'><div class='cites-label'>Quy định cụ thể</div>"]
 
  for citation in citations:
    used = citation['doc_id'] in used_doc_ids
 
    article, document_name = split_title(citation['title'])
 
    passage = pipeline.corpus_store.passage_for_generator(citation['doc_id'], 'context')
 
    seal = "<span class='cite-seal'>đã dẫn</span>" if used else ""
 
    blocks.append(
        f"<details class='cite{' used' if used else ''}'{' open' if used else ''}>"
        f"<summary>"
        f"<span class='cite-chevron'>▶</span>"
        f"<span class='cite-title'>{html.escape(article)}"
        f"<span class='cite-doc'>{html.escape(document_name)}</span></span>"
        f"{seal}"
        f"</summary>"
        f"<div class='cite-body'>{to_html(passage)}</div>"
        f"</details>"
    )
 
  blocks.append("</div>")
 
  st.markdown("".join(blocks), unsafe_allow_html=True)
 
 
def render_answer(question: str, context: str) -> str:

  from transformers import TextIteratorStreamer
 
  generator = pipeline.generator
  tokenizer = generator.tokenizer
 
  tokenizer.truncation_side = 'left'
 
  encoded = tokenizer(
      [generator.render_prompt(question, context)], return_tensors='pt',
      truncation=True, max_length=generator.config['max_source_length'],
      add_special_tokens=False,
  ).to(generator.model.device)
 
  streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
 
  thread = Thread(target=generator.model.generate, kwargs=dict(
      **encoded, streamer=streamer,
      max_new_tokens=generator.config['max_target_length'],
      num_beams=generator.config.get('num_beams', 1),
      do_sample=generator.config.get('do_sample', False),
      pad_token_id=tokenizer.pad_token_id,
      eos_token_id=tokenizer.eos_token_id,
  ))
 
  thread.start()
 
  placeholder = st.empty()
 
  partial = ""
 
  for chunk in streamer:
    partial += chunk
 
    placeholder.markdown(
        bubble('assistant', rag.Generator.clean_generation(partial) or "…"),
        unsafe_allow_html=True,
    )
 
  thread.join()
 
  answer = rag.Generator.clean_generation(partial)
 
  placeholder.markdown(bubble('assistant', answer), unsafe_allow_html=True)
 
  return answer
 
 
# --- Thanh tiêu đề 
st.markdown(
    "<div class='app-bar'>"
    "<div class='app-mark'>🎓️</div>"
    "<div><div class='app-name'>Hệ thống tư vấn quy về định đào tạo UIT</div>"
    "<div class='app-note'>Mỗi câu trả lời kèm điều luật gốc để bạn đối chiếu</div></div>"
    "</div>",
    unsafe_allow_html=True,
)
 
SUGGESTIONS = [
    "Sinh viên bị cảnh báo học vụ mấy lần thì bị buộc thôi học?",
    "Điều kiện xét tốt nghiệp loại xuất sắc là gì?",
    "Chuẩn ngoại ngữ đầu ra của sinh viên là bao nhiêu?",
    "Một học kỳ được đăng ký tối đa bao nhiêu tín chỉ?",
]
 
if not conversation['turns']:
  st.markdown(
      "<div class='empty'>"
      "<div class='empty-mark'>🎓️</div>"
      "<h1>Bạn cần tra quy định nào?</h1>"
      "<p>Hỏi về quy chế học vụ, đăng ký môn học, điều kiện xét tốt nghiệp "
      "hay chuẩn ngoại ngữ đầu ra?</p>"
      "<div class='empty-hint'>Thử hỏi</div>"
      "</div>",
      unsafe_allow_html=True,
  )
 
  left_column, right_column = st.columns(2, gap='small')
 
  for index, suggestion in enumerate(SUGGESTIONS):
    with (left_column if index % 2 == 0 else right_column):
      if st.button(suggestion, key=f'suggest_{index}', use_container_width=True):
        st.session_state.pending_question = suggestion
        st.rerun()
 
for turn in conversation['turns']:
  st.markdown(bubble('user', turn['question']), unsafe_allow_html=True)
  st.markdown(bubble('assistant', turn['answer']), unsafe_allow_html=True)
 
  render_citations(turn['citations'], turn['used_doc_ids'])
 
question = st.chat_input("Nhập câu hỏi mà bạn muốn hỏi về quy định đào tạo…")
 
if 'pending_question' in st.session_state:
  question = st.session_state.pop('pending_question')
 
if question:
  question = rag.normalize_text(question)
 
  st.markdown(bubble('user', question), unsafe_allow_html=True)
 
  with st.spinner("Đang tra điều luật…"):
    ranking = pipeline.retrieve(question).head(settings.DISPLAY_K)
 
  used_doc_ids = ranking['doc_id'].tolist()[:settings.TOP_K]
 
  context = pipeline.corpus_store.build_generator_context(
      used_doc_ids,
      passage_column=pipeline.generator.config['passage_column'],
      separator=pipeline.generator.config.get('context_separator', '\n\n'),
  )
 
  answer = render_answer(question, context)
 
  citations = [{'doc_id': row.doc_id, 'title': row.title, 'score': float(row.score)}
               for row in ranking.itertuples(index=False)]
 
  render_citations(citations, used_doc_ids)
 
  conversation['turns'].append({
      'question': question, 'answer': answer,
      'citations': citations, 'used_doc_ids': used_doc_ids,
  })
 
  # Tên cuộc trò chuyện lấy từ câu hỏi đầu tiên.
  if len(conversation['turns']) == 1:
    conversation['title'] = question[:60]
 
  save_conversations()