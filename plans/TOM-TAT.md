# Phân bổ tính toán thị giác theo loại suy luận cho VQA tiếng Việt

*Tóm tắt nội bộ — Paper 3, Trust4NLP @ ACIIDS 2027*

## Tóm tắt

Các mô hình VQA thưa như ViMoE-VQA giả định loại suy luận của câu hỏi là tín hiệu hữu ích
để phân bổ tính toán đa phương thức. Chúng tôi kiểm tra trực tiếp giả định này. Trên nền
một VLM đóng băng hoàn toàn (InternViT-300M + Qwen2-0.5B) chỉ huấn luyện một bridge nhẹ,
chúng tôi định nghĩa không gian hành động rời rạc trên (số tile ảnh × kiến trúc bridge) và
quét oracle: với mỗi câu hỏi, đánh giá mọi hành động một lần để có hành động tối ưu và
khoảng cách chất lượng. Kết quả: loại suy luận không dự đoán nhu cầu tính toán thị giác ở
nhóm câu hỏi nào; dư địa per-sample của oracle là nhiễu CIDEr, không chuyển giao giữa các
tập; và không policy học được nào — dùng loại suy luận, đặc trưng thị giác rẻ, hay cả hai
— vượt được policy cố định "bridge tốt nhất, ít tile nhất". Kèm theo một kết quả dương:
bridge multi-token (0,78% tham số) đạt CIDEr-D 94,4, vượt ViMoE-VQA và mọi baseline đã
công bố trên các chỉ số sinh.

## 1. Giới thiệu

Chi phí suy diễn của VLM tập trung ở bộ mã hoá thị giác; điều chỉnh ngân sách thị giác
theo từng câu hỏi là hướng tự nhiên, và câu hỏi mở là tín hiệu nào điều khiển quyết định.
ViMoE-VQA ngầm giả định loại suy luận là tín hiệu đó. Chúng tôi kiểm tra bằng oracle trên
AutoViVQA: một router có giám sát loại suy luận có phân bổ tính toán tốt hơn router chỉ
dùng đặc trưng thị giác rẻ, hay tốt hơn policy cố định, không? Câu trả lời là không, ở cả
ba mức — theo nhóm câu hỏi, per-sample, và so với policy cố định.

## 2. Kiến trúc

**Mô hình nền.** Vintern-1B-v3.5 theo mô hình encoder–bridge–decoder: bộ mã hoá thị giác
InternViT-300M và bộ giải mã ngôn ngữ Qwen2-0.5B, cả hai **đóng băng**. Chúng tôi thay
projector tuyến tính gốc bằng một **bridge** huấn luyện được, ánh xạ các patch token của
InternViT sang không gian embedding của Qwen2. Chỉ bridge được tối ưu, bằng cross-entropy
trên câu trả lời tham chiếu; InternViT và Qwen2 giữ nguyên trọng số pretrained.

Ảnh được cắt thành `n_tiles` tile 448×448 (tiling động kiểu InternVL: chọn lưới khớp tỉ
lệ ảnh, cộng một thumbnail toàn ảnh); mỗi tile qua InternViT cho 256 patch token. Bridge
nhận `n_tiles × 256` token và xuất ra `k` vision token đưa vào Qwen2 cùng token câu hỏi.

**Năm bridge**, trải trên hai trục thiết kế pooled/attentive và ít/nhiều token:

| Bridge | k token | Cơ chế | Tham số huấn luyện |
|---|---:|---|---:|
| Residual | 1 | projector tuyến tính + nhánh residual LayerNorm/GELU | 4,86M (0,52%) |
| Multi-token | 8 | gộp trung bình patch → 8 token (1 anchor + 7 ngữ nghĩa) | 7,35M (0,78%) |
| Tile-attention | 8 | self-attention giữa patch rồi gộp | 4,14M (0,44%) |
| Light q-former | 8 | 8 query, cross-attention 2 lớp | 27,6M (2,87%) |
| Full q-former | 16 | 16 query, 4 lớp, fusion ảnh–văn bản | 69,4M (6,91%) |

**Router** chạy song song với bộ mã hoá thị giác và rẻ hơn nhiều, sinh hai tín hiệu.
P(r|Q) là bộ phân loại PhoBERT-base tám lớp trên loại suy luận, **chỉ nhìn câu hỏi**, đạt
macro-F1 kiểm định 0,91. f(I,Q) là đặc trưng thị giác rẻ, không cần mã hoá đa tile: token
CLS của InternViT ở một tile (giảm chiều PCA còn 64), độ dài câu hỏi, và ba điểm chất
lượng ảnh cấp mẫu (độ nét, che khuất, mật độ vật thể).

**Không gian hành động.** Một hành động `a = (n_tiles, bridge)` với `n_tiles ∈ {1, 3, 6}`
và `bridge` thuộc ba bridge mạnh nhất (multi-token, full q-former, light q-former), cho
`|A| = 9`. Chi phí `C(a) = n_tiles / 6 ∈ (0, 1]` — số lần chạy InternViT, chuẩn hoá.
Profiling P1 (P100, Vintern-1B thật) xác nhận `n_tiles` là đòn bẩy tính toán thật: từ 1
lên 6 tile, InternViT tốn ×6,0 FLOPs, độ trễ đầu-cuối ×4,0, thông lượng ×5,2. Chất lượng
`M(a; x)` là CIDEr per-sample của câu trả lời.

**Oracle và policy.** Với hệ số đánh đổi `λ`, hành động oracle là
`a*(x, λ) = argmax_a [M(a; x) − λ C(a)]`; quét `λ ∈ {0, 0.05, 0.1, 0.2, 0.4, 0.7, 1}`.
Lượt quét oracle đánh giá **toàn bộ 9 hành động trên mọi câu hỏi** (train/kiểm định/test),
lưu `M` và `C`. Policy `π_θ(P(r|Q), f(I,Q), λ) → a` là một MLP nhỏ, huấn luyện bằng phân
loại có giám sát với nhãn `a*` trên tập train. Các nhánh ablation khác nhau duy nhất ở đầu
vào: `ours` (đủ), `rt_only` (chỉ loại suy luận), `visual_only` (chỉ đặc trưng thị giác);
đường tham chiếu gồm mọi hành động cố định, chọn ngẫu nhiên, và oracle.

## 3. Dữ liệu

**AutoViVQA** (Nguyen và cộng sự, 2026) gồm 19.411 ảnh MS-COCO, 37.077 câu hỏi tiếng Việt
và 185.385 câu trả lời (5 câu trả lời tự do mỗi câu hỏi, 1–10 token), kèm nhãn loại suy
luận. Bộ dữ liệu chỉ phát hành phân chia train/kiểm định 80/20, **không có tập test công
khai**.

**Phân chia lại (grouped 70/15/15).** Chúng tôi gán mỗi ảnh vào đúng một tập theo hash
`image_id`, seed 42, giữ xấp xỉ phân bố loại suy luận. Không ảnh — do đó không caption hay
bối cảnh chung ảnh — xuất hiện ở hai tập; đây là kênh rò rỉ mà phân chia ngẫu nhiên cấp
câu hỏi để hở.

| Tập | Câu hỏi | Ảnh | Độ dài câu hỏi | Độ dài câu trả lời |
|---|---:|---:|---:|---:|
| Train | 25.933 | 13.576 | 11,4 từ | 4,3 từ |
| Kiểm định | 5.544 | 2.908 | 11,4 từ | 4,3 từ |
| Test | 5.503 | 2.914 | 11,4 từ | 4,3 từ |

Trùng ảnh giữa hai tập bất kỳ: **0**.

**Phân bố loại suy luận** (tập train, ổn định giữa các tập): relational 30%, recognition
19%, spatial 15%, causal 13%, counting 12%, action 7%, context 3%, yes/no 1%. Trường
`reason_depth` (Level 1–5) được giữ để tham chiếu nhưng dùng như nhãn danh nghĩa —
Level 5 chỉ ~200 mẫu train, không phải bậc "khó nhất" đáng tin.

**Tập con oracle.** Lượt quét oracle (9 hành động × mọi câu hỏi) đắt, nên chạy trên một
tập con cân bằng theo loại (giới hạn ~625 câu mỗi loại): 5.547 câu train, 3.727 kiểm định,
3.739 test. Đây là sai lệch có chủ đích so với phân tầng theo tỉ lệ; thống kê tổng thể
được tái cân bằng về phân bố tự nhiên.

## 4. Kết quả

### 4.1 So sánh bridge và baseline

Chỉ số tính bằng thư viện token-metric in-house (nhất quán với bảng kết quả AutoViVQA
trước), quy ước ×100, trên tập kiểm định, seed 42. Baseline lấy từ công trình đã công bố.

| Mô hình | Acc | Prec | Rec | F1 | BLEU | ROUGE | METEOR | CIDEr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Vintern-1B (base) | 0,12 | 17,52 | 19,87 | 17,55 | 1,91 | 25,84 | 23,93 | 8,54 |
| ViT5+ViT | 7,97 | 46,84 | 50,33 | 48,52 | 4,13 | 46,89 | 31,02 | 72,68 |
| BARTPhoBEiT | 8,81 | 45,30 | 46,48 | 45,88 | 4,33 | 44,83 | 24,57 | 188,96 |
| Vintern-1B (finetune toàn bộ) | **13,01** | 52,47 | 55,12 | 53,76 | 6,11 | **51,93** | 35,25 | 72,84 |
| Llama 3.2 (zero-shot) | 0,36 | 23,96 | 73,71 | 36,16 | 3,62 | 36,11 | 30,01 | 62,84 |
| Gemini 2.0 Flash (zero-shot) | 0,55 | 27,20 | 74,10 | 39,79 | 4,41 | 39,60 | 31,72 | 74,42 |
| Gemini 2.5 Flash (zero-shot) | 0,22 | 24,43 | 76,66 | 24,75 | 0,39 | 37,27 | 31,22 | 71,90 |
| GPT-5 (zero-shot) | 10,84 | 47,20 | 55,20 | 50,89 | 6,07 | 47,30 | 33,34 | 84,20 |
| ViMoE-VQA / Tuong-MoE (5 seed) | 9,65 | **62,89** | 58,65 | **60,69** | 12,54 | 47,07 | **39,10** | 88,67 |
| **Residual bridge** | 1,87 | 34,57 | 43,70 | 36,45 | 6,11 | 34,12 | 30,33 | 66,07 |
| **Multi-token bridge** | 8,62 | 51,60 | 52,32 | 50,66 | **16,34** | 48,95 | 41,05 | **98,69** |
| **Tile-attention bridge** | 6,06 | 47,11 | 49,13 | 46,69 | 13,52 | 44,91 | 37,36 | 87,46 |
| **Light q-former** | 5,99 | 47,18 | 49,00 | 46,63 | 13,80 | 44,81 | 37,30 | 88,10 |
| **Full q-former** | 7,34 | 48,31 | 49,78 | 47,66 | 14,58 | 45,96 | 38,25 | 90,82 |

Multi-token là bridge mạnh nhất trên mọi chỉ số dù ít tham số hơn q-former một bậc độ
lớn; residual (một token) thua xa — biểu diễn ảnh bằng một vector là ràng buộc chính, chứ
không phải độ sâu bridge. Multi-token vượt ViMoE-VQA trên BLEU (+3,8), ROUGE (+1,9), CIDEr
(+10,0) nhưng thua F1 và Precision; chúng tôi quy điều này cho bộ giải mã Qwen2-0.5B đóng
băng (ViMoE huấn luyện decoder sáu lớp từ đầu với label smoothing). Với các chỉ số ổn
định giữa các implementation (pycocoevalcap cấp corpus), multi-token đạt CIDEr-D 94,4 /
BLEU-4 19,6 / ROUGE-L 50,0, vẫn vượt ViMoE-VQA (88,7 / 12,5 / 47,1); METEOR biến thiên
~13 điểm giữa các implementation nên không dùng để so sánh liên công trình.

### 4.2 Phân tích oracle

**(a) Theo nhóm câu hỏi: `n_tiles` không có tác dụng.** Với hai bridge cross-attention
(q-former, light q-former), CIDEr trung bình theo (nhóm × n_tiles) không cho nhóm nào mà
tăng tile giúp có ý nghĩa: bootstrap ghép cặp per-sample cho hiệu số n3−n1 và n6−n1 với
khoảng tin cậy 95% chứa 0 ở cả tám nhóm (tập kiểm định, 3.727 mẫu). Một pilot 591 mẫu
từng gợi ý spatial/context/recognition lợi +0,11–0,14 CIDEr; hiệu ứng này biến mất khi
quét toàn tập — chỉ là nhiễu mẫu nhỏ. Kết luận: **loại suy luận không dự đoán nhu cầu
tính toán thị giác.**

**(b) Per-sample, oracle trông vượt xa hành động cố định.** Trên tập test `|A| = 9`
(3.739 mẫu), oracle `a*(x, 0)` đạt CIDEr trung bình 1,26 so với 0,90 của hành động cố
định tốt nhất `multi_token|t1` — chênh +0,36 (+40%). Khoảng chênh này trải đều trên cả
tám nhóm (+0,28 đến +0,65 mỗi nhóm), không tập trung ở một nhóm suy luận nào.

**(c) Nhưng +40% đó là nhiễu đo lường, không phải cấu trúc học được.** Ba quan sát:

1. *Đầu vào đủ mạnh để biểu diễn a\*.* Một policy huấn luyện **và** đánh giá trên cùng
   tập test tái tạo gần đúng oracle (khớp a* 0,98, CIDEr 1,26 — không phân biệt được với
   cận trên). Vậy `(P(r|Q), f(I,Q))` không thiếu thông tin; vấn đề nằm ở nhãn a*.
2. *a\* không chuyển giao.* Phân bố hành động tối ưu trên tập kiểm định và tập test khác
   hẳn nhau dù phân chia và phân tầng giống hệt: hành động a* đa số là `qformer|t1` trên
   kiểm định nhưng `multi_token|t1` trên test. Policy học a* của tập này không dùng được
   cho tập kia.
3. *Nguyên nhân.* Chín hành động cho chất lượng gần bằng nhau (chênh thật ~0,05 CIDEr),
   trong khi CIDEr per-sample trên một câu trả lời bốn từ dao động lớn hơn thế nhiều.
   argmax trên chín giá trị gần bằng nhau vì vậy bị phương sai đo lường chi phối —
   oracle đang "chọn trúng" mẫu ngẫu nhiên cao điểm, không phải mẫu thật sự tốt hơn.
   Cận trên oracle 1,25 cũng bị thổi phồng bởi cùng nhiễu này.

### 4.3 Ablation policy (|A| = 9, tập test tách biệt)

| Nhánh | khớp a* | CIDEr TB | chi phí TB |
|---|---:|---:|---:|
| Oracle | 1,00 | 1,25 | 0,29 |
| **Cố định `multi_token\|t1`** | — | **0,902** | 0,167 |
| ours | 0,43 | 0,901 | 0,168 |
| rt_only | 0,44 | 0,902 | 0,167 |
| visual_only | 0,44 | 0,82 | 0,225 |
| Ngẫu nhiên | — | 0,77 | 0,56 |

Với đủ dữ liệu huấn luyện (5.547 câu), cả ba nhánh policy hội tụ đúng về hành động cố định
tốt nhất, độc lập với `λ`, tỉ lệ khớp a* xấp xỉ tỉ lệ lớp đa số. Huấn luyện trên tập nhỏ
hơn thì các policy này overfit và tụt xuống dưới baseline cố định. Không gian `|A| = 6`
cho bức tranh y hệt.

## 5. Thảo luận

Đòn bẩy tính toán thị giác là thật (chi phí ×4–6) nhưng phẳng về chất lượng: bộ giải mã
0,5 tỉ tham số đóng băng không chuyển được chi tiết thị giác thêm vào thành câu trả lời
tốt hơn, ở mọi nhóm suy luận — nên không router nào, dù được cung cấp loại suy luận hay
không, thêm được giá trị so với "luôn dùng cấu hình tốt nhất". Kết quả không mâu thuẫn với
số của ViMoE-VQA nhưng gợi ý lợi ích của MoE nên được quy cho dung lượng và ensemble hơn
là định tuyến theo loại suy luận; ablation leave-one-out của chính ViMoE cho thấy các
expert của nó không chuyên biệt hoá mạnh.

Ba đóng góp đứng độc lập: một khung instrumentation (VLM chỉ-train-bridge, không gian hành
động có chi phí chuẩn hoá, oracle offline) để nghiên cứu phân bổ tính toán thị giác; một
kết quả âm được thiết lập chặt chẽ cho chế độ frozen-backbone, kèm truy nguyên nguyên
nhân; và một benchmark bridge không rò rỉ trên AutoViVQA cùng phân tích chi phí tính toán
mà công trình trước bỏ ngỏ.

**Hạn chế.** Kết quả hiện ở một seed cho các lần chạy bridge và policy; cần giao thức năm
seed hoặc lập luận bootstrap ghép cặp. Các bridge huấn luyện ở một tile, dưới cấu hình
thị giác thiết kế của Vintern (tới 12 tile) — baseline Vintern-finetune chạy đa tile, nên
khoảng cách F1 lẫn ảnh hưởng của tile; một lần retrain multi-token có tăng cường số tile
đang chạy để kiểm tra kết luận có đổi không. Chưa có kiểm định của người và phân tích lỗi
định lượng.
