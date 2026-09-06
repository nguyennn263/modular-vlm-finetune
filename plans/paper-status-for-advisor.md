# Báo cáo tiến độ nghiên cứu — Paper 3

*Cập nhật ngày 06/09/2026.*
*Tài liệu chi tiết (đầy đủ 7 bảng và 2 hình, kèm phụ lục theo từng seed):*
*`plans/paper-blueprint.md`.*

---

## 1. Câu hỏi nghiên cứu

> Thay vì xây dựng một mô hình mới (như ViMoE-VQA), liệu có thể cải thiện
> Vintern-1B trên tập dữ liệu AutoViVQA bằng cách chỉ cập nhật một phần nhỏ tham
> số (khoảng 1% tổng số, đóng băng toàn bộ backbone) mà vẫn đạt kết quả tương
> đương với fine-tune hay không? Nếu chưa đạt, điểm nghẽn (bottleneck) nằm ở đâu?

**Vị trí của nghiên cứu so với các công trình liên quan:**

| Công trình | Phương pháp | Tham số được cập nhật |
|---|---|---|
| Vintern-1B (fine-tuned) | Fine-tune toàn bộ InternViT-300M và projector; áp dụng LoRA cho Qwen2-0.5B | Phần lớn tham số phía thị giác và projector |
| ViMoE-VQA | Thiết kế kiến trúc Mixture-of-Experts mới | Toàn bộ mô hình mới |
| **Nghiên cứu này** | Đóng băng đồng thời InternViT-300M và Qwen2-0.5B; chỉ huấn luyện bridge (0.78%) và LoRA cho decoder (0.23%), dùng 1 tile | **Khoảng 1% tổng số tham số** |

---

## 2. Phát hiện chính hiện tại

Trong phạm vi các can thiệp đã khảo sát, việc tăng dung lượng (capacity) ở phía
thị giác — bridge lớn hơn, tăng số lượng tile, hoặc định tuyến (routing) thích
ứng — không mang lại cải thiện đáng kể. Trục điều chỉnh phía decoder là hướng duy
nhất cho thấy tín hiệu cải thiện rõ ràng trong các thí nghiệm hiện có. Từ đó,
phương pháp đề xuất bao gồm: một bridge dạng pooling chi phí thấp (cố định) kết
hợp với LoRA nhẹ cho decoder.

Phương pháp này đạt hoặc vượt Vintern-1B fine-tuned trên các chỉ số sinh văn bản
(BLEU +14.9, METEOR +10.0, CIDEr +36.8), trong khi chỉ cập nhật khoảng 1% tổng số
tham số và giữ đóng băng toàn bộ backbone. So với ViMoE-VQA, phương pháp cao hơn
ở BLEU, ROUGE, METEOR và CIDEr, nhưng thấp hơn ở token-F1 (−6.0). Khoảng cách F1
này là động lực cho phần phân tích điểm nghẽn.

---

## 3. Kết quả chính (tập validation, chỉ số đánh giá nội bộ, thang ×100)

| Mô hình | Acc | Prec | Rec | F1 | BLEU | ROUGE | METEOR | CIDEr |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Vintern-1B (gốc, zero-shot) | 0.12 | 17.52 | 19.87 | 17.55 | 1.91 | 25.84 | 23.93 | 8.54 |
| Vintern-1B (fine-tuned) | 13.01 | 52.47 | 55.12 | 53.76 | 6.11 | 51.93 | 35.25 | 72.84 |
| GPT-5 (zero-shot) | 10.84 | 47.20 | 55.20 | 50.89 | 6.07 | 47.30 | 33.34 | 84.20 |
| ViMoE-VQA | 9.65 | 62.89 | 58.65 | 60.69 | 12.54 | 47.07 | 39.10 | 88.67 |
| **Bridge Multi-Token (0.78%, 1 tile)** | **8.28** | **50.53** | **51.72** | **49.82** | **15.99** | **48.11** | **40.47** | **96.98** |
| **  + LoRA cho decoder, r=16 (~1.0%)** | **10.42** | **53.85** | **55.00** | **53.17** | **19.44** | **51.48** | **43.91** | **105.59** |
| **  + LoRA cho decoder, r=16, 3 epoch** | **11.78** | **55.54** | **56.25** | **54.67** | **20.98** | **52.92** | **45.24** | **109.60** |

*(Bảng đầy đủ chín baseline, so sánh chỉ số corpus với ViMoE và khoảng tin cậy:
xem `paper-blueprint.md`, Bảng 1–2.)*

Theo cách đo corpus (dùng để so sánh với các công trình khác): Bridge Multi-Token
đạt CIDEr-D 92.8 ± 1.1 (khoảng tin cậy 95%: [91.3, 97.1], nằm hoàn toàn trên mức
88.7 của ViMoE); cấu hình kết hợp LoRA và 3 epoch đạt 106.8 ± 1.1. Hạn chế còn
lại: token-F1 và Acc vẫn thấp hơn ViMoE.

---

## 4. Phân tích điểm nghẽn — sáu trục can thiệp, một trục cho kết quả tích cực

Giá trị ΔF1 được tính so với cấu hình gốc (Bridge Multi-Token, seed 42: F1
50.66):

| Câu hỏi · trục | Can thiệp | ΔF1 | Nhận xét |
|---|---|--:|---|
| RQ1–2 · Dung lượng của bridge | Full Q-Former (69M tham số, gấp 10 lần) | −3.00 | Tăng dung lượng bridge không cải thiện kết quả trong khảo sát này |
| RQ3 · Số lượng tile thị giác | Huấn luyện với 1 tile → đánh giá với 3 tile | −29.61 | Bridge huấn luyện với đầu vào một tile suy giảm mạnh khi suy luận với nhiều tile ᵃ |
| RQ4 · Định tuyến thích ứng | Chính sách (policy) học theo loại câu hỏi | ≈0 | Không vượt trội so với cấu hình cố định |
| RQ5 · Tín hiệu huấn luyện | Lấy mẫu nhiều câu trả lời tham chiếu | −1.65 | Không cải thiện |
| RQ5 · Căn chỉnh biểu diễn | Chưng cất kiến thức (KD) tại projector, mức đặc trưng | −1.00 | Không cải thiện |
| **RQ6 · Dung lượng của decoder** | **LoRA r=16 (1 epoch)** | **+2.51** | **Cải thiện nhất quán** |
| **RQ6 · Dung lượng của decoder** | **LoRA r=16 (3 epoch)** | **+4.01** | **Cải thiện nhất quán** |

ᵃ Thí nghiệm này huấn luyện với 1 tile và đánh giá với 3–6 tile, do đó chỉ cho
phép kết luận rằng bridge huấn luyện với đầu vào một tile có khả năng khái quát
hóa kém sang chế độ suy luận nhiều tile; chưa thể kết luận về hiệu quả của việc
huấn luyện đa tile.

**Nhận định:** Hiệu ứng tích cực của LoRA cho decoder xuất hiện nhất quán trên
mọi loại bridge, cho thấy đây không phải là kết quả ngẫu nhiên của một cấu hình
cụ thể; trong khi đó, cả bốn trục can thiệp phía thị giác đều không cho tín hiệu
cải thiện. Các kết quả hiện tại cho thấy frozen decoder là một điểm nghẽn đáng
kể, còn việc tăng dung lượng ở bridge không mang lại lợi ích tương ứng.

Hai hình minh họa (xem `paper-blueprint.md`):

- **Hình 1 — Hiện tượng đồng đều hóa giữa các bridge:** các bridge ở cấu hình gốc
  trải rộng CIDEr-D từ 56 đến 97; sau khi bổ sung LoRA cho decoder (0.23% tham
  số), tất cả đều hội tụ về khoảng 100–107.
- **Hình 2 — Suy giảm khi tăng số tile:** F1 giảm từ 50.7 xuống 21 khi tăng số
  tile đánh giá từ 1 lên 3 (bridge được huấn luyện với 1 tile); validation loss
  tăng từ 1.48 lên 3.36.

---

## 5. Độ tin cậy của kết quả

- Bridge Multi-Token (cấu hình gốc): trung bình trên 4 seed.
- Bridge Multi-Token kết hợp LoRA r=16 (1 epoch và 3 epoch), và Q-Former kết hợp
  LoRA r=16: trung bình trên 3 seed.
- Các cấu hình còn lại trong Mục 4 hiện dựa trên 1 seed; đang được bổ sung để đạt
  3 seed.
- Phương án grouped split đã được kiểm chứng: kết quả của các bridge gần như
  không thay đổi so với cách chia cũ, cho thấy các kết quả trước đó không bị
  phóng đại do rò rỉ dữ liệu.
- Mọi phép so sánh chính đều kèm khoảng tin cậy bootstrap.
- Đánh giá ngữ nghĩa hiện ở mức tự kiểm tra trên 120 mẫu với một người đánh giá.

---

## 6. Đóng góp của bài báo

1. **Quy trình thích nghi tiết kiệm tham số (parameter-efficient adaptation
   recipe):** đóng băng phần thị giác, kết hợp một bridge nhẹ và LoRA cho decoder;
   chỉ khoảng 1% tham số được huấn luyện nhưng đạt hoặc vượt baseline fine-tuned
   trên các chỉ số sinh văn bản.
2. **Phương pháp chẩn đoán điểm nghẽn một cách hệ thống (systematic bottleneck
   diagnosis):** khảo sát có hệ thống dung lượng bridge, mở rộng số tile, định
   tuyến, tín hiệu giám sát và căn chỉnh biểu diễn, cùng với thích nghi decoder;
   kết quả cho thấy thích nghi decoder là hướng duy nhất mang lại cải thiện nhất
   quán trong không gian can thiệp đã khảo sát.
3. **Quy trình đánh giá đáng tin cậy (reliable evaluation protocol):** grouped
   split chống rò rỉ dữ liệu, đánh giá trên nhiều seed, khoảng tin cậy bootstrap,
   đánh giá thủ công kèm phân tích lỗi; đồng thời phân tích hiệu quả tính toán
   của tham số số tile.
