# Error analysis — residual (val, seed 42, 5463 samples)

- Mean predicted length **6.41** tokens vs reference **4.32**; prediction shorter than the mean reference in **33.4%** of cases.

## Token-F1 buckets (max over 5 refs)
| bucket | % |
|---|---:|
| strong (F1>=.6) | 17.1 |
| partial (.2-.6) | 57.9 |
| weak (0-.2) | 8.5 |
| zero (F1=0) | 16.5 |

## Per reasoning-type
| category | n | mean token-F1 |
|---|---:|---:|
| relational | 1662 | 0.379 |
| recognition | 1016 | 0.351 |
| spatial | 802 | 0.311 |
| causal | 692 | 0.259 |
| counting | 689 | 0.526 |
| action | 391 | 0.294 |
| context | 145 | 0.273 |
| yesno | 66 | 0.493 |

## Counting questions: noun omission = **8.4%** (58/689) — cf. ViMoE-VQA 10.7%.

## Examples — zero token-F1
- *[causal]* Q: Tại sao một số người tham gia bị bịt mắt?
  - pred: **Bịt mắt do bệnh lý mắt hoặc do tác động lực lượng.**  | refs: ['Để tăng độ khó', 'Đó là quy tắc cuộc thi', 'Để tạo bất ngờ', 'Là một phần của trò chơi', 'Để công bằng hơn']
- *[relational]* Q: Mối quan hệ giữa người trượt ván và người xem là gì?
  - pred: **Cảm xúc hài hước**  | refs: ['Người biểu diễn và khán giả', 'Vận động viên và người hâm mộ', 'Người trượt ván và người quan sát', 'Màn trình diễn và người theo dõi', 'Đối tượng được chiêm ngưỡng']
- *[context]* Q: Người chơi tennis có thể đang nghĩ gì khi thi đấu?
  - pred: **Hình ảnh là một người đang chơi tennis**  | refs: ['Làm sao để thắng trận', 'Tập trung vào cú đánh tiếp theo', 'Cố gắng hết sức mình', 'Vượt qua đối thủ', 'Giành điểm số quan trọng']
- *[causal]* Q: Mục đích của việc quay video những người trượt ván đang thực hiện thủ thuật là gì?
  - pred: **Hướng dẫn người tham gia**  | refs: ['Để ghi lại thủ thuật', 'Để xem lại sau này', 'Để tạo kỷ niệm', 'Có thể để chia sẻ', 'Để học hỏi động tác']
- *[context]* Q: Người lính đứng trong đống đổ nát có thể đang cảm thấy gì?
  - pred: **Cảm thấy vất vả**  | refs: ['Có lẽ anh ta đang thất vọng', 'Có lẽ anh ta đang buồn', 'Có lẽ anh ta đang sốc', 'Có lẽ anh ta đang mệt mỏi', 'Có lẽ anh ta đang tuyệt vọng']
- *[causal]* Q: Tại sao mấy con chó trông không vui vẻ khi lướt sóng?
  - pred: **Một con chó đang chạy**  | refs: ['Chúng không thích lướt sóng', 'Chúng sợ nước', 'Có thể do sóng lớn', 'Chúng không thoải mái', 'Do cảm thấy bất an']
