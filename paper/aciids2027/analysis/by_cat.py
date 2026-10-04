import json, sys, glob, collections, statistics as st
sys.path.insert(0, '.')
from metrics.compute_score import preprocess_sentence, normalize_text
from metrics.f1 import F1
f1 = F1()
def wf1(gts, p):
    pp = " ".join(preprocess_sentence(normalize_text(p)))
    return max(f1.compute_score({'0': [" ".join(preprocess_sentence(normalize_text(g)))]}, {'0': [pp]})[0] for g in gts)
S = '/private/tmp/claude-501/-Users-tcx-Documents-personal-Repo-modular-vlm-finetune/81a4e167-edde-416f-9614-4b31c737b49b/scratchpad/diag_out'
val = [json.loads(l) for l in open('data/splits/val.jsonl')]
def load(tag):
    f = glob.glob(f'{S}/{tag}/**/text_predictions_epoch_1.json', recursive=True)[0]
    s = json.load(open(f))['samples']; assert len(s) == len(val)
    for a, b in zip(s, val): assert a['question'] == b['question']
    return s
plain, lora = load('mt-s42-t1-full'), load('l3ep-s42-t1-full')
by = collections.defaultdict(lambda: [[], []])
for p, l, v in zip(plain, lora, val):
    by[v['category']][0].append(wf1(p['ground_truths'], p['prediction']))
    by[v['category']][1].append(wf1(l['ground_truths'], l['prediction']))
print(f"{'category':14s} {'n':>5s} {'plain':>6s} {'lora':>6s} {'delta':>6s}")
for c, (a, b) in sorted(by.items(), key=lambda x: -len(x[1][0])):
    print(f"{c:14s} {len(a):5d} {100*st.mean(a):6.2f} {100*st.mean(b):6.2f} {100*(st.mean(b)-st.mean(a)):+6.2f}")
allp = [x for a, _ in by.values() for x in a]; alll = [x for _, b in by.values() for x in b]
print(f"{'ALL':14s} {len(allp):5d} {100*st.mean(allp):6.2f} {100*st.mean(alll):6.2f} {100*(st.mean(alll)-st.mean(allp)):+6.2f}")
json.dump({'plain': [p['prediction'] for p in plain], 'lora': [l['prediction'] for l in lora]}, open(S + '/preds_s42_full.json', 'w'), ensure_ascii=False)
