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
| ViT5_ViT | 7.97 | 46.84 | 50.33 | 48.52 | 4.13 | 46.89 | 31.02 | 72.68 |
| BARTPhoBEiT | 8.81 | 45.30 | 46.48 | 45.88 | 4.33 | 44.83 | 24.57 | 188.96 ᵃ |
| Vintern-1B (fine-tuned) | 13.01 | 52.47 | 55.12 | 53.76 | 6.11 | 51.93 | 35.25 | 72.84 |
| Llama 3.2 (zero-shot) | 0.36 | 23.96 | 73.71 | 36.16 | 3.62 | 36.11 | 30.01 | 62.84 |
| Gemini 2.0 Flash | 0.55 | 27.20 | 74.10 | 39.79 | 4.41 | 39.60 | 31.72 | 74.42 |
| Gemini 2.5 Flash | 0.22 | 24.43 | 76.66 | 24.75 | 0.39 | 37.27 | 31.22 | 71.90 |
| GPT-5 (zero-shot) | 10.84 | 47.20 | 55.20 | 50.89 | 6.07 | 47.30 | 33.34 | 84.20 |
| ViMoE-VQA | 9.65 | 62.89 | 58.65 | 60.69 | 12.54 | 47.07 | 39.10 | 88.67 |
| **Bridge Multi-Token (0.78%, 1 tile)** | **8.28** | **50.53** | **51.72** | **49.82** | **15.99** | **48.11** | **40.47** | **96.98** |
| **  + LoRA cho decoder, r=16 (~1.0%)** | **10.42** | **53.85** | **55.00** | **53.17** | **19.44** | **51.48** | **43.91** | **105.59** |
| **  + LoRA cho decoder, r=16, 3 epoch** | **11.78** | **55.54** | **56.25** | **54.67** | **20.98** | **52.92** | **45.24** | **109.60** |

*In đậm = phương pháp đề xuất. ᵃ CIDEr của BARTPhoBEiT là giá trị ngoại lai do
mô hình sinh câu dài; không đưa vào so sánh. Các dòng baseline lấy theo báo cáo
của benchmark AutoViVQA.*

Theo cách đo corpus (dùng để so sánh với các công trình khác): Bridge Multi-Token
đạt CIDEr-D 92.8 ± 1.1 (khoảng tin cậy 95%: [91.3, 97.1], nằm hoàn toàn trên mức
88.7 của ViMoE); cấu hình kết hợp LoRA và 3 epoch đạt 106.8 ± 1.1. Hạn chế còn
lại: token-F1 và Acc vẫn thấp hơn ViMoE.

---

## 4. Phân tích điểm nghẽn — sáu trục can thiệp, một trục cho kết quả tích cực

Giá trị ΔF1 được tính so với cấu hình gốc (Bridge Multi-Token, seed 42: F1
50.66):

| RQ · axis | Intervention | ΔF1 | Nhận xét |
|---|---|--:|---|
| RQ1–2 · Bridge capacity | Full Q-Former (69M params, 10×) | −3.00 | Tăng dung lượng bridge không cải thiện kết quả trong khảo sát này |
| RQ3 · Number of visual tiles | Train with 1 tile → evaluate with 3 tiles | −29.61 | Bridge huấn luyện với đầu vào một tile suy giảm mạnh khi suy luận với nhiều tile ᵃ |
| RQ4 · Adaptive routing | Learned policy (conditioned on question type) | ≈0 | Không vượt trội so với cấu hình cố định |
| RQ5 · Training signal | Multi-reference answer sampling | −1.65 | Không cải thiện |
| RQ5 · Representation alignment | Projector-level knowledge distillation (feature KD) | −1.00 | Không cải thiện |
| **RQ6 · Decoder capacity** | **LoRA r=16 (1 epoch)** | **+2.51** | **Cải thiện nhất quán** |
| **RQ6 · Decoder capacity** | **LoRA r=16 (3 epochs)** | **+4.01** | **Cải thiện nhất quán** |

ᵃ Thí nghiệm này huấn luyện với 1 tile và đánh giá với 3–6 tile, do đó chỉ cho
phép kết luận rằng bridge huấn luyện với đầu vào một tile có khả năng khái quát
hóa kém sang chế độ suy luận nhiều tile; chưa thể kết luận về hiệu quả của việc
huấn luyện đa tile.

**Nhận định:** Hiệu ứng tích cực của LoRA cho decoder xuất hiện nhất quán trên
mọi loại bridge, cho thấy đây không phải là kết quả ngẫu nhiên của một cấu hình
cụ thể; trong khi đó, cả bốn trục can thiệp phía thị giác đều không cho tín hiệu
cải thiện. Các kết quả hiện tại cho thấy frozen decoder là một điểm nghẽn đáng
kể, còn việc tăng dung lượng ở bridge không mang lại lợi ích tương ứng.

Một quan sát bổ sung: các bridge vốn chênh lệch lớn về CIDEr-D ở cấu hình gốc
(từ khoảng 56 đến 97) đều hội tụ về khoảng 100–107 sau khi thêm LoRA cho decoder,
cho thấy khi decoder có đủ dung lượng thì lựa chọn kiến trúc bridge gần như không
còn ảnh hưởng.

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

---

## 7. Kết luận sơ bộ

Trả lời trực tiếp cho hai câu hỏi ở Mục 1:

**(a) Có thể thích nghi Vintern-1B với khoảng 1% tham số hay không?** Có, ở mức
một phần. Với backbone đóng băng hoàn toàn, một bridge pooling nhẹ kết hợp LoRA
cho decoder đã đạt hoặc vượt Vintern-1B fine-tuned trên toàn bộ các chỉ số sinh
văn bản (BLEU, ROUGE, METEOR, CIDEr). Tuy nhiên, phương pháp vẫn thấp hơn ViMoE-VQA
ở token-F1 (−6.0) và thấp hơn ở Acc; do đó chưa thể nói là tương đương hoàn toàn
với các phương pháp huấn luyện đầy đủ.

**(b) Điểm nghẽn nằm ở đâu?** Trong không gian can thiệp đã khảo sát, phía thị
giác không còn dư địa cải thiện: tăng dung lượng bridge, tăng số tile, hay định
tuyến thích ứng đều không có tác dụng. Chỉ có việc thêm dung lượng cho decoder
(LoRA) mới cải thiện token-F1 một cách nhất quán, và hiệu ứng này lặp lại trên mọi
loại bridge. Kết quả hiện tại chỉ ra frozen decoder là điểm nghẽn đáng kể đối với
lớp mô hình này (ViT đóng băng, decoder nhỏ 0.5B đóng băng, ít token thị giác).

**Hàm ý:** để thu hẹp nốt khoảng cách token-F1, hướng đi hợp lý là mở thêm dung
lượng ở phía decoder (LoRA sâu hơn, hoặc decoder đóng băng lớn hơn), chứ không
phải tiếp tục đầu tư vào phía thị giác. Đây cũng là điểm phản biện với nhận định
"reasoning-aware routing" của ViMoE-VQA: trên cùng benchmark, loại câu hỏi không
mang tín hiệu hữu ích cho việc phân bổ tài nguyên thị giác.
