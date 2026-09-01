

Vintern-1B được thiết kế lightweight:
Projector đơn giản chỉ với MLP 2 lớp làm nhiệm vụ: vision embedding → LLM embedding space


Giữ cố định mô hình thị giác và mô hình ngôn ngữ của Vintern-1B, chỉ huấn luyện thành phần bridge module để ánh xạ đặc trưng ảnh sang không gian embedding của mô hình ngôn ngữ. 5 kiến trúc bridge khác nhau được đề xuất nhằm cải thiện lớp linear project gốc.

Exp 1 – Residual Bridge: mở rộng bridge tuyến tính bằng cách bổ sung nhánh residual gồm LayerNorm và hai lớp fully-connected với GELU, giúp học phần hiệu chỉnh trên biểu diễn gốc.
	
Exp 2 – Multi-Token Bridge: thay vì biểu diễn ảnh bằng một token duy nhất, mô hình tạo nhiều token đầu ra, trong đó một token giữ vai trò anchor và các token còn lại bổ sung thông tin ngữ nghĩa.
Exp 3 – Tile Attention Bridge: chia đặc trưng ảnh thành nhiều patch và áp dụng self-attention giữa các patch để tận dụng thông tin không gian.
Exp 4 – Lightweight Q-Former: sử dụng 8 query token cùng 2 lớp transformer nhẹ để học tương tác giữa đặc trưng thị giác và không gian embedding ngôn ngữ.
Exp 5 – Full Q-Former: mở rộng kiến trúc Q-Former với 16 query token, 4 lớp transformer và cơ chế fusion giữa thông tin ảnh và văn bản, nhằm tăng khả năng căn chỉnh đa phương thức.

Mô hình
Acc
Prec
Rec
F1
BLEU
ROUGE
METEOR
CIDEr
Vintern (base)
0.12
17.52
19.87
17.55
1.91
25.84
23.93
8.54
ViT5_ViT
7.97
46.84
50.33
48.52
4.13
46.89
31.02
72.68
BARTPhoBEiT
8.81
45.30
46.48
45.88
4.33
44.83
24.57
188.96
Vintern (finetune)
13.01
52.47
55.12
53.76
6.11
51.93
35.25
72.84
Llama 3.2
0.36
23.96
73.71
36.16
3.62
36.11
30.01
62.84
Gemini 2.0 Flash
0.55
27.20
74.10
39.79
4.41
39.60
31.72
74.42
Gemini 2.5 Flash
0.22
24.43
76.66
24.75
0.39
37.27
31.22
71.90
GPT-5
10.84
47.20
55.20
50.89
6.07
47.30
33.34
84.20
Tuong-MOE
9.65
62.89
58.65
60.69
12.54
47.07
39.10
88.67
Residual Bridge
1.27
34.18
38.50
33.09
3.46
29.66
25.18
58.24
Multi-Token Bridge
8.69
50.14
53.19
50.23
16.47
48.25
41.55
99.88
Tile Attention Bridge
7.99
48.82
51.23
48.60
14.59
46.83
39.74
96.03
Lightweight Q-Former
7.61
48.32
49.09
47.28
13.95
45.50
37.72
91.32
Full Q-Former
7.15
48.63
50.38
48.09
14.82
46.29
38.98
94.17





Qua thực nghiệm có thể rút ra ba nhận xét chính:
Việc thay đổi riêng tầng bridge của Vintern-1B có thể tạo ra cải thiện đáng kể mà không cần huấn luyện lại toàn bộ mô hình.
Các kiến trúc sử dụng nhiều token ảnh (Multi-Token, Tile Attention, QForm) hiệu quả hơn đáng kể so với cách chỉ tinh chỉnh MLP một token (Residual Bridge).
Nhìn chung, cả ba mô hình cải tiến đều chưa vượt qua Vintern-1B gốc về chỉ số F1, Tuy nhiên, hai phương pháp sử dụng nhiều token đầu ra là Multi-Token Bridge,Tile Attention Bridge, QForm cho thấy khả năng cải thiện rõ rệt ở các chỉ số đánh giá sinh văn bản như BLEU và CIDEr. Điều này cho thấy hướng mở rộng bridge từ biểu diễn một token sang nhiều token là tiềm năng cho việc nâng cao chất lượng trả lời trong bài toán VQA..
Multi-Token Bridge là phương pháp tốt nhất trong các cải tiến đề xuất, đạt BLEU, METEOR và CIDEr cao nhất.
Residual Bridge
2026-06-10 09:31:13,979 - data_loader_logger - INFO - Total parameters: 939,484,161
2026-06-10 09:31:13,979 - data_loader_logger - INFO - Trainable parameters: 4,855,553 (0.52%)
2026-06-10 09:31:13,979 - data_loader_logger - INFO - Frozen parameters: 934,628,608


Multi-Token Bridge
2026-05-02 02:16:15,471 - data_loader_logger - INFO - Total parameters: 941,975,808
2026-05-02 02:16:15,471 - data_loader_logger - INFO - Trainable parameters: 7,347,200 (0.78%)
2026-05-02 02:16:15,471 - data_loader_logger - INFO - Frozen parameters: 934,628,608

Tile Attention Bridge
2026-06-10 09:31:54,531 - data_loader_logger - INFO - Total parameters: 938,770,816
2026-06-10 09:31:54,531 - data_loader_logger - INFO - Trainable parameters: 4,142,208 (0.44%)
2026-06-10 09:31:54,531 - data_loader_logger - INFO - Frozen parameters: 934,628,608

Lightweight Q-Former
2026-06-10 09:33:26,474 - data_loader_logger - INFO - Total parameters: 962,195,840
2026-06-10 09:33:26,474 - data_loader_logger - INFO - Trainable parameters: 27,567,232 (2.87%)
2026-06-10 09:33:26,474 - data_loader_logger - INFO - Frozen parameters: 934,628,608

Full Q-Former
2026-06-10 09:33:48,700 - data_loader_logger - INFO - Total parameters: 1,004,015,744
2026-06-10 09:33:48,700 - data_loader_logger - INFO - Trainable parameters: 69,387,136 (6.91%)
2026-06-10 09:33:48,700 - data_loader_logger - INFO - Frozen parameters: 934,628,608

