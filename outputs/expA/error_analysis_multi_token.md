# Error analysis — multi_token (val, seed 42, 5463 samples)

- Mean predicted length **4.39** tokens vs reference **4.32**; prediction shorter than the mean reference in **49.6%** of cases.

## Token-F1 buckets (max over 5 refs)
| bucket | % |
|---|---:|
| strong (F1>=.6) | 36.7 |
| partial (.2-.6) | 51.5 |
| weak (0-.2) | 3.1 |
| zero (F1=0) | 8.7 |

## Per reasoning-type
| category | n | mean token-F1 |
|---|---:|---:|
| relational | 1662 | 0.551 |
| recognition | 1016 | 0.453 |
| spatial | 802 | 0.476 |
| causal | 692 | 0.425 |
| counting | 689 | 0.66 |
| action | 391 | 0.396 |
| context | 145 | 0.368 |
| yesno | 66 | 0.614 |

## Counting questions: noun omission = **5.8%** (40/689) — cf. ViMoE-VQA 10.7%.

## Examples — zero token-F1
- *[causal]* Q: Tại sao một số người tham gia bị bịt mắt?
  - pred: **Vì trời mưa**  | refs: ['Để tăng độ khó', 'Đó là quy tắc cuộc thi', 'Để tạo bất ngờ', 'Là một phần của trò chơi', 'Để công bằng hơn']
- *[context]* Q: Người chơi tennis có thể đang nghĩ gì khi thi đấu?
  - pred: **Đang suy nghĩ về kết quả**  | refs: ['Làm sao để thắng trận', 'Tập trung vào cú đánh tiếp theo', 'Cố gắng hết sức mình', 'Vượt qua đối thủ', 'Giành điểm số quan trọng']
- *[causal]* Q: Tại sao một trong những người trượt ván lại nhảy lên không trung?
  - pred: **Có thể do không khí**  | refs: ['Để thực hiện một thủ thuật', 'Để khoe kỹ năng của anh ấy', 'Để thực hiện động tác trượt ván', 'Anh ấy đang biểu diễn', 'Thực hiện một pha nguy hiểm']
- *[context]* Q: Những người đang xem người đàn ông lướt sóng có thể đang cảm thấy gì?
  - pred: **Cảm thấy vui vẻ**  | refs: ['Có lẽ họ đang thích thú', 'Có thể họ đang ghen tị', 'Họ có thể đang học hỏi', 'Có thể họ đang chờ lượt', 'Có thể họ đang thư giãn']
- *[action]* Q: Đàn bò có vẻ đang làm gì gần bờ sông?
  - pred: **Đang ăn thức ăn**  | refs: ['Đứng nghỉ ngơi', 'Uống nước', 'Chờ đợi', 'Thư giãn', 'Gần gũi con người']
- *[causal]* Q: Tại sao tóc trên đầu của vận động viên trượt ván lại đứng vững?
  - pred: **Để giữ vững ván**  | refs: ['Có lẽ do tốc độ nhanh.', 'Có thể vì gió mạnh khi trượt.', 'Do tác động của không khí.', 'Biểu hiện của sự phấn khích.', 'Tóc dựng lên tự nhiên khi vận động.']
