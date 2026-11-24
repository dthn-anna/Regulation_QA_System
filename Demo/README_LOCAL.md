# Chuyển hệ thống về máy local

## Phần 1 — Lấy những file gì từ Drive

Gốc trên Drive: `/content/drive/MyDrive/Legal_Sytem/`

### Bắt buộc — không có thì hệ thống không chạy

| Trên Drive | Về local | Dung lượng |
|---|---|---|
| `01_Dataset/02_processed_data/corpus/corpus.parquet` | `data/corpus.parquet` | vài MB |
| `02_Retriever/03_Finetuning/models/{best_run}/` (cả thư mục) | `models/retriever/bge-m3/` | ~2,2 GB |
| `03_Generator/models_1/{RUN_NAME}/` (cả thư mục) | `models/generator/Qwen3-0.6B/` | ~1,2 GB |
| `02_Retriever/serving/retriever_serving.json` | `config/retriever_serving.json` | vài KB |

Thư mục generator phải có `prompt_config.json` bên trong — sinh bởi mục 08 của
`03__Generator_train`. Checkpoint train bằng bản cũ sẽ không có file này.

Đường dẫn chính xác của checkpoint retriever là giá trị `MODEL_SAVE_DIR` trong
notebook 4c; mở `{best_run}__config.json` để đọc.

### Nên có — để kiểm chứng và hiển thị dashboard

| Trên Drive | Về local | Dùng làm gì |
|---|---|---|
| `01_Dataset/02_processed_data/corpus/qa_master.parquet` | `data/qa_master.parquet` | `verify_local.py` và hiệu chỉnh ngưỡng từ chối |
| `02_Retriever/04_Hybrid/output/hybrid__*__test_ranking.parquet` | `data/reference/` | đối chiếu retriever local với thực nghiệm |
| `05_End2End/output/*__test_predictions.parquet` (lấy một) | `data/reference/` | đối chiếu generator local |
| `05_End2End/output/end2end_matrix__test.csv` | `data/reference/` | tab Thực nghiệm |
| `05_End2End/output/error_decomposition__test.csv` | `data/reference/` | tab Thực nghiệm |
| `05_End2End/output/behaviour__test.csv` | `data/reference/` | tab Thực nghiệm |
| `05_End2End/output/significance__test.csv` | `data/reference/` | tab Thực nghiệm |

### KHÔNG lấy

`train.xlsx` / `val.xlsx` / `test.xlsx`, `queries.parquet`, `qrels.parquet`,
`qa_pair.parquet`, `answer.parquet`, toàn bộ `03_Finetuning/train_data/`
(`train_examples`, `train_pairs_flat`, `teacher_scores`), các file ranking của
BM25 và dense không dùng làm đối chiếu.

Đặc biệt **đừng lấy `checkpoints_1/`** — đó là checkpoint trung gian của Trainer,
mỗi epoch một bản, có thể vài chục GB. Bạn chỉ cần `models_1/`.

### Nén trước khi tải

Drive tải thư mục nhiều file rất chậm. Trên Colab:

```python
!cd /content/drive/MyDrive/Legal_Sytem && \
 tar -czf /content/system_bundle.tar.gz \
   01_Dataset/02_processed_data/corpus/corpus.parquet \
   01_Dataset/02_processed_data/corpus/qa_master.parquet \
   02_Retriever/serving/retriever_serving.json \
   02_Retriever/03_Finetuning/models/bge-m3 \
   03_Generator/models_1/Qwen3-0.6B \
   05_End2End/output

from google.colab import files
files.download('/content/system_bundle.tar.gz')
```

---

## Phần 2 — Thiết kế hệ thống local

```
legal-qa/
├── settings.py              # mọi đường dẫn và tham số phục vụ
├── rag_pipeline.py          # lõi: corpus, retriever, generator, pipeline
├── retriever_common.py      # copy từ Drive, rag_pipeline gọi vào
├── app_streamlit.py         # giao diện
├── verify_local.py          # kiểm chứng local == thực nghiệm
├── requirements.txt
│
├── config/retriever_serving.json
├── data/corpus.parquet, qa_master.parquet, reference/
├── models/retriever/bge-m3/, generator/Qwen3-0.6B/
└── cache/                   # tự sinh: vector corpus
```

### Trách nhiệm từng file

| File | Giữ cái gì | Ai gọi |
|---|---|---|
| `settings.py` | đường dẫn, `TOP_K`, ngưỡng từ chối, thiết bị | tất cả |
| `rag_pipeline.py` | `CorpusStore`, `BM25Retriever`, `DenseRetriever`, `HybridRetriever`, `Generator`, `RAGPipeline` | app + verify |
| `retriever_common.py` | mã hoá corpus, max-pool chunk — dùng chung với notebook | `rag_pipeline` |
| `app_streamlit.py` | chỉ giao diện, không có logic truy hồi | người dùng |
| `verify_local.py` | đối chiếu với kết quả Colab, hiệu chỉnh ngưỡng | chạy tay |

Nguyên tắc chính: **`app_streamlit.py` không được chứa tham số mô hình nào.**
Mọi thứ ảnh hưởng tới kết quả nằm trong hai file JSON và được `rag_pipeline` đọc.
Giao diện chỉ hiển thị. Khi cần đổi retriever hay generator, bạn sửa
`settings.py`, không đụng vào app.

### Chạy

```bash
pip install -r requirements.txt

python settings.py          # liệt kê file còn thiếu
python verify_local.py      # chứng minh local == thực nghiệm
streamlit run app_streamlit.py
```

### Ba tab của dashboard

**Hỏi đáp** — chat có streaming, mỗi câu trả lời kèm điều luật gốc mở ra được,
đánh dấu điều luật nào thực sự được dùng để sinh.

**Thực nghiệm** — bốn bảng từ notebook 05: ma trận end-to-end, phân rã lỗi, hành
vi khi context sai, kiểm định thống kê. Đọc từ `data/reference/`.

**Hệ thống** — hai khối JSON cấu hình đang chạy. Đây là phần trả lời trực tiếp
câu hỏi "làm sao biết demo giống thực nghiệm": mọi tham số hiện ra màn hình đều
đọc từ file sinh lúc thực nghiệm, không cái nào gõ tay.

### Vì sao `TOP_K` không cho chỉnh trên giao diện

Generator được fine-tune với đúng một điều luật cho mỗi câu hỏi. Cho người dùng
kéo thanh trượt lên 5 là đưa mô hình ra ngoài phân bố huấn luyện, và số ở tab
Thực nghiệm không còn mô tả cái đang chạy nữa. Thanh trượt duy nhất trong sidebar
là số điều luật *hiển thị*, không ảnh hưởng tới prompt.

---

## Phần 3 — Những chỗ local dễ lệch khỏi Colab

**`pyvi` khác phiên bản** là nguyên nhân số một. `ViTokenizer` tách từ khác thì
chỉ mục BM25 khác, điểm khác, thứ hạng khác — và không có exception nào.
`pip freeze | grep pyvi` trên Colab rồi ghim đúng số đó.

**Quên copy checkpoint retriever.** `SentenceTransformer` sẽ lặng lẽ tải model
gốc từ HuggingFace. Chạy được, kém hơn hẳn, không báo lỗi. `verify_local.py`
bắt được vì ranking không khớp.

**Không có GPU.** Model chạy `float32` thay `bfloat16`, greedy decoding lệch vài
mẫu — dấu phẩy động, không phải sai định dạng. Qwen3-0.6B trên CPU sinh khoảng
5–15 token/giây, một câu trả lời 10–30 giây. bge-m3 mã hoá 294 passage trên CPU
mất vài phút, nên `build_index` có cache, chỉ chạy lần đầu.

**Cache vector cũ.** Cache đặt tên theo vân tay của checkpoint, toàn bộ tham số
ảnh hưởng tới vector, và danh sách `doc_id`. Đổi bất cứ thứ gì là sinh cache mới.

**Đường dẫn trong `retriever_serving.json`.** Khoá `model_save_dir` vẫn trỏ về
Drive. Sửa lại, hoặc để `settings.DENSE_MODEL_DIR` ghi đè — `RAGPipeline.load`
ưu tiên tham số đó.
