"""Does the trained Full Q-Former exploit the answer it can see in its text input?
Same forward as training, two conditions for the Q-Former's text input:
  A) full input_ids (question + ANSWER)  -- what training does
  B) question part only (up to answer_start_pos)
The LLM always receives the full sequence; loss = CE on answer tokens only."""
import sys, statistics as st, torch, torch.nn.functional as F
sys.path.insert(0, '.')
from transformers import AutoModel, AutoTokenizer
from src.config.loader import load_config, repo_root
from src.training.setup import create_finetune_model
from src.data.split import load_split
from src.data.collator import create_collate_fn
torch.set_grad_enabled(False)
name = "5CD-AI/Vintern-1B-v3_5"
cfg = load_config(repo_root() / "configs/bridges/qformer.yaml")
base = AutoModel.from_pretrained(name, torch_dtype=torch.float32, trust_remote_code=True).eval()
model = create_finetune_model(base, bridge_type=cfg["bridge_type"], bridge_config=cfg.get("bridge_config") or {}).eval().float()
ck = torch.load("checkpoints/expA/seed42/qformer/last_model.pt", map_location="cpu", weights_only=False)
model.bridge.load_state_dict(ck.get("bridge_state", ck))
tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
collate = create_collate_fn(tokenizer=tok, image_size=(336, 336), max_length=256)
emb = model.language_model.get_input_embeddings()
samples = load_split("val", "data/splits")[:24]
res = {"A_full": [], "B_question_only": []}
for s in samples:
    b = collate([s]); n = int(b["attention_mask"][0].sum()); asp = int(b["answer_start_pos"][0])
    ids = b["input_ids"][:, :n]
    vis = model.vision_model(b["pixel_values"].float()).last_hidden_state      # (1, 577, 1024)
    text = emb(ids).float()
    for key, qf_text in [("A_full", text), ("B_question_only", text[:, :asp])]:
        br = model.bridge(vis, qf_text)
        out = model.language_model(inputs_embeds=torch.cat([br, text], 1)).logits[:, br.shape[1]:]
        # position t-1 predicts token t, for answer tokens t in [asp, n)
        loss = F.cross_entropy(out[0, asp - 1:n - 1], ids[0, asp:n])
        res[key].append(loss.item())
for k, v in res.items():
    print(f"{k:16s} answer-token CE = {st.mean(v):.3f}  (n={len(v)})")
d = [a - b for a, b in zip(res["A_full"], res["B_question_only"])]
print(f"A lower than B on {sum(x < 0 for x in d)}/{len(d)} questions; mean diff {st.mean(d):+.3f}")
