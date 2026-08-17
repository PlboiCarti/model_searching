# Kế hoạch tiếp theo: image-only end-to-end retrieval

## Mục tiêu và ranh giới

Mục tiêu là nhận `clip_queries` tiếng Anh từ repo xử lý query, dùng CLIP
ViT-B/32 + FAISS để lấy keyframe, fuse các query, rồi refine trên video gốc để
trả về `original frame_id`.

```text
Repo query
  QueryPlan -> clip_queries + query weights
  -> search_clip_queries(...)
  -> Weighted RRF
  -> temporal regions
  -> refine_regions(...)
  -> task-specific output

model_searching
  FAISS keyframe retrieval + CLIP text/image encoder + raw-video refinement
```

Baseline này chỉ dùng image encoder. Không thêm OCR, ASR, caption, Qdrant, hay
training vào phạm vi hiện tại.

## Phase 0 - Xác thực artifact của đồng đội

**Chủ yếu ở `model_searching`; hoàn thành trước khi nối hai repo.**

1. Copy một bundle hoàn chỉnh và cấu hình `.env`:
   `AIC_ARTIFACT_DIR`, `AIC_CLIP_MODEL_NAME=ViT-B/32`, `AIC_USE_LORA`.
2. Kiểm tra bundle có `video.index`, `index_metadata.json`, và nếu dùng LoRA
   thì có `lora_weights.pt` cùng checkpoint family.
3. Kiểm tra từng metadata row có tối thiểu:
   `video_id`, `frame_id` thật, `pts_time`, `keyframe_ordinal`; để refine cần
   thêm `original_video_path` hoặc một cách map chắc chắn sang video gốc.
4. Chạy một truy vấn tiếng Anh thủ công qua `search_clip_queries` và kiểm tra
   kết quả map đúng video/keyframe/frame ID.

**Tiêu chí xong:** index/metadata có cùng số phần tử, query vector cùng số
chiều với index, và top results có `frame_id` thật chứ không phải filename
keyframe.

## Phase 1 - Đóng gói retrieval thành dependency cục bộ

Hai repository không thể tự import lẫn nhau. Không merge source; biến
`model_searching` thành Python package có tên riêng, ví dụ
`aic-model-searching` / `aic_model_searching`.

1. Thêm `pyproject.toml` và public import tối thiểu cho
   `search_clip_queries`, các result dataclass, và exception retrieval.
2. Không dùng `from backend...` từ repo query: tên `backend` quá chung và có
   thể đụng package của repo query. Public API phải dùng namespace riêng:

   ```python
   from aic_model_searching import search_clip_queries
   ```

3. Trong virtual environment của repo query, cài local editable dependency:

   ```powershell
   python -m pip install -e D:\VideoQuery\model_searching
   ```

   Editable install tạo liên kết đến source hiện tại, không copy code và không
   cần merge. Mọi sửa đổi ở `model_searching` được repo query dùng ngay.
4. Repo query và package retrieval phải cùng thấy `.env`/environment variables
   trỏ đến bundle artifact (`AIC_ARTIFACT_DIR`) và video gốc khi refinement.
5. Không dùng HTTP service trong baseline local. Chỉ cân nhắc service khi hai
   repo chạy ở máy hoặc môi trường tách biệt.

**Tiêu chí xong:** một script ở repo query import được
`aic_model_searching`, gọi `search_clip_queries`, và nhận
`list[QueryRetrievalResult]` mà không thay đổi hay copy source retrieval.

## Phase 2 - Nối retrieval vào repo query

**Thực hiện ở repo query. Không sửa logic LLM trong `model_searching`.**

1. Gọi `search_clip_queries(clip_queries, top_k=N)` bằng các final English
   queries do planner tạo ra.
2. Giữ nguyên `QueryRetrievalResult(query_index, query_text, candidates)`;
   không flatten mất provenance.
3. Khai báo weight theo vai trò query ở repo query, ví dụ anchor `1.0`, visual
   expansion `0.7-0.9`. Một query đơn thì weight `1.0` và không cần RRF.
4. Viết `weighted_rrf(per_query_results, query_weights, k=60)`:

   ```text
   fused_score(video_id, frame_id)
     = sum(weight_q / (60 + rank_q))
   ```

5. Lưu provenance: score, rank và các query đã hỗ trợ mỗi fused keyframe.

**Tiêu chí xong:** một query có nhiều `clip_queries` trả ra một ranking chung;
không cộng trực tiếp raw CLIP cosine scores giữa các query.

## Phase 3 - Gom fused keyframe thành temporal region

**Thực hiện ở repo query, ngay sau RRF.**

1. Sắp `fused candidates` theo `video_id`, rồi theo `pts_time`.
2. Chỉ gom các frame thuộc cùng video và sát nhau theo một `max_gap_sec` cấu
   hình được. Mỗi region giữ:

   ```text
   video_id, start_pts, end_pts, peak_candidate,
   member_candidates, supporting_query_indices, fused_score
   ```

3. `peak_candidate` là keyframe có fused score cao nhất; region không thay
   thế frame ID, nó chỉ là cửa sổ để decode video một lần.
4. Chọn Top-M regions đa dạng theo video/thời gian để gửi sang refinement.

**Tiêu chí xong:** ba keyframe gần nhau của cùng cảnh chỉ tạo một region; ba
keyframe ở video khác hoặc cách xa nhau vẫn là region độc lập.

## Phase 4 - Dense refinement trên video gốc

**Thực hiện ở `model_searching`. Đây là phần code mới cần viết.**

1. Thêm `backend/refinement.py` với public API dự kiến:

   ```python
   refine_regions(regions, weighted_clip_queries) -> list[RefinedFrame]
   ```

2. `RefinedFrame` cần trả: `video_id`, `frame_id`, `pts_time`, `score`,
   `source_region`, và provenance query.
3. Mở `original_video_path` quanh region. Refine theo hai tầng để không encode
   cả corpus:

   - quét thưa trong cửa sổ region có padding;
   - chọn local peak rồi quét mọi frame quanh peak.

4. Encode những raw frames đó bằng **cùng CLIP image encoder ViT-B/32**. Text
   query dùng cùng text encoder/optional text-only LoRA đang dùng cho FAISS.
5. Dùng PTS/frame mapping thật khi xuất `frame_id`; không suy `frame_id` bằng
   `time * fps` cho video VFR. Tái sử dụng logic ffprobe/map-keyframes hiện có
   hoặc đóng gói nó thành helper read-only.
6. Viết test cho: video path thiếu, region rỗng, mapping PTS/frame, và chọn
   đúng peak trên fixture video nhỏ.

**Tiêu chí xong:** một region coarse trả ra một `original frame_id` refine;
refinement chỉ decode các cửa sổ Top-M chứ không re-index toàn bộ video.

## Phase 5 - Task output trong repo query

1. **Textual KIS:** fused regions -> refined frames -> diversify Top-100.
2. **QA:** dùng refined evidence frame; phần trả lời là module riêng, không
   nằm trong retrieval backend.
3. **TRAKE:** mỗi event search/refine riêng; không RRF lẫn các event. Sau đó
   dùng DP để chọn cùng video và frame ID tăng dần.
4. Giữ CSV/evaluator ở lớp ngoài cùng; chỉ dùng format BTC chính thức khi đã
   được xác nhận.

## Thứ tự thực hiện ngay bây giờ

1. Hoàn thành Phase 0 với artifact thật của đồng đội.
2. Đóng gói theo Phase 1 rồi kiểm tra repo query gọi được retrieval.
3. Bạn triển khai Phase 2 và Phase 3 trong repo query.
4. Sau khi có `FusedRegion` contract mẫu, quay lại repo này để triển khai
   Phase 4 đúng input/output đó.
5. Chạy ba query đại diện end-to-end trước khi mở rộng sang task-specific
   logic và submission.
