# Error analysis — qformer (val, seed 42, 5463 samples)

- Mean predicted length **4.39** tokens vs reference **4.32**; prediction shorter than the mean reference in **48.6%** of cases.

## Token-F1 buckets (max over 5 refs)
| bucket | % |
|---|---:|
| strong (F1>=.6) | 32.8 |
| partial (.2-.6) | 53.0 |
| weak (0-.2) | 3.3 |
| zero (F1=0) | 10.9 |

## Per reasoning-type
| category | n | mean token-F1 |
|---|---:|---:|
| relational | 1662 | 0.521 |
| recognition | 1016 | 0.412 |
| spatial | 802 | 0.434 |
| causal | 692 | 0.4 |
| counting | 689 | 0.648 |
| action | 391 | 0.393 |
| context | 145 | 0.324 |
| yesno | 66 | 0.518 |

## Counting questions: noun omission = **5.2%** (36/689) — cf. ViMoE-VQA 10.7%.

## Examples — zero token-F1
- *[context]* Q: Người lính đứng trong đống đổ nát có thể đang cảm thấy gì?
  - pred: **Cảm thấy đau đớn**  | refs: ['Có lẽ anh ta đang thất vọng', 'Có lẽ anh ta đang buồn', 'Có lẽ anh ta đang sốc', 'Có lẽ anh ta đang mệt mỏi', 'Có lẽ anh ta đang tuyệt vọng']
- *[causal]* Q: Tại sao mấy con chó trông không vui vẻ khi lướt sóng?
  - pred: **Vì họ đang bị trượt**  | refs: ['Chúng không thích lướt sóng', 'Chúng sợ nước', 'Có thể do sóng lớn', 'Chúng không thoải mái', 'Do cảm thấy bất an']
- *[causal]* Q: Tại sao người đàn ông và phụ nữ lại mỉm cười?
  - pred: **Để tạo dáng**  | refs: ['Vì họ đang nâng cốc chúc mừng', 'Có thể họ đang vui vẻ', 'Vì họ đang thưởng thức rượu', 'Có lẽ họ đang trò chuyện vui vẻ', 'Vì họ đang có khoảnh khắc đẹp']
- *[action]* Q: Đàn bò có vẻ đang làm gì gần bờ sông?
  - pred: **Đang quay lưng về phía máy ảnh**  | refs: ['Đứng nghỉ ngơi', 'Uống nước', 'Chờ đợi', 'Thư giãn', 'Gần gũi con người']
- *[relational]* Q: Mối quan hệ giữa hai cô gái trẻ trong trận đấu là gì?
  - pred: **Cô gái bên trái đang tấn công**  | refs: ['Đối thủ', 'Đồng đội', 'Người chơi', 'Vận động viên', 'Bạn chơi']
- *[relational]* Q: Mối quan hệ giữa việc chơi thể thao và tình bạn thân thiết là gì?
  - pred: **Chúng được kết nối**  | refs: ['Thể thao giúp xây dựng tình bạn', 'Thể thao dạy tình bạn thân thiết', 'Tình bạn phát triển qua thể thao', 'Chơi cùng nhau tạo tình bạn', 'Thể thao là nền tảng tình bạn']
