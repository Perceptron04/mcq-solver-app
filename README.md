# Smart MCQ Solver

Ranks the three most likely answers for a five-option multiple-choice
question, using a DeBERTa-v3 model fine-tuned with LoRA.

**Live demo:** https://mcq-solver-app-deep-learning.streamlit.app/

**Model:** [Tokyo0412/mcq-solver-deberta](https://huggingface.co/Tokyo0412/mcq-solver-deberta)

Sagar K Chaudhary · 23f2002523 · BS Data Science, IIT Madras

---

## What it does

Enter a question and five candidate answers. The model scores each option
and returns the top three in ranked order, along with the confidence for
all five.

Scoring in the original competition used MAP@3, which awards 1.0 if the
correct answer is ranked first, 0.5 if second and 0.333 if third.

---

## Results

| | Value |
|---|---|
| Final leaderboard score | **0.75644** |
| Qualifying cutoff | 0.73 |
| Random baseline | 0.3578 |
| Validation MAP@3 | 0.9983 |
| Trainable parameters | 1,476,097 of 185,899,010 (0.79%) |

The competition submission blends this DeBERTa model with an ELECTRA
model at a weight of 0.25. The demo runs the DeBERTa half alone, since
loading two transformers exceeds the memory available on free hosting.

---

## How it works

Each option is paired with the question and passed through the model as a
single sequence, so attention can compare their words directly:

The model emits one score per option, softmax converts those into
probabilities, and the top three are returned.

This joint encoding is what separates the approach from a similarity
baseline. Encoding the question and option independently and comparing
the resulting vectors scored 0.2567 accuracy; reading them together
reached 0.9967.

**Fine-tuning.** LoRA freezes all 186 million pretrained weights and
trains small low-rank adapters beside the attention layers instead. A
768 by 768 weight update, which would hold 589,824 values, is replaced by
two thin matrices holding 24,576 — about 96% fewer. Only 0.79% of the
model was trained.

---

## Running locally

```bash
git clone https://github.com/Perceptron04/mcq-solver-app
cd mcq-solver-app
pip install -r requirements.txt
streamlit run app.py
```

The first launch downloads the model from HuggingFace, which takes a
minute. After that it is cached.

---

## Files
app.py Streamlit application
requirements.txt pinned dependencies
.streamlit/ config (disables the file watcher)

---

## Notes on deployment

The model is 870 MB in float32. The app loads it with
`low_cpu_mem_usage=True` so the weights stream in shard by shard rather
than being duplicated in memory, then quantizes the linear layers to int8
at load time. That brings the footprint down substantially, which matters
on a host with limited memory.

---

## A caveat worth stating

The competition dataset contains an annotation artifact: the correct
answers are written in a subtly different style from the distractors, so
a model that never sees the question at all still scores 100% on
validation against a 20% random baseline. I verified this on the
leaderboard too — a submission built purely on that pattern scored
0.75311.

I then measured how much this model depends on the same shortcut, and it
agrees with the question-blind detector on 99.7% of validation rows. So
its accuracy on this dataset does not fully transfer to questions written
differently. This is documented in full in the project report.
