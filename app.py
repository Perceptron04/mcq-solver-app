"""
Smart MCQ Solver — Streamlit demo
Sagar K Chaudhary | 23f2002523

Ranks the three most likely answers for a five-option multiple-choice
question using a DeBERTa-v3 model fine-tuned with LoRA.
"""

import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForMultipleChoice

# ---------------------------------------------------------------- config
REPO = "Tokyo0412/mcq-solver-deberta"
OPTS = ["A", "B", "C", "D", "E"]
MAX_LEN = 256

st.set_page_config(page_title="Smart MCQ Solver", page_icon="?", layout="centered")


# ---------------------------------------------------------------- model
@st.cache_resource(show_spinner="Loading model, this takes a minute on first run...")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(REPO)

    # float16 halves the memory: ~740 MB becomes ~370 MB.
    # low_cpu_mem_usage loads shard by shard so there is never a
    # second full copy in RAM at once.
    model = AutoModelForMultipleChoice.from_pretrained(
        REPO,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model = model.float()      # back to float32 for CPU inference
    model.eval()

    try:
        model = torch.ao.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8)
    except Exception:
        pass

    return tokenizer, model


def predict(question, options, tokenizer, model):
    """Score five options and return probabilities in A-E order.

    The question is repeated five times and paired with each option, so
    the tokenizer produces [CLS] question [SEP] option [SEP] for each.
    That is the same format the model was trained on.
    """
    enc = tokenizer(
        [question] * 5, options,
        truncation=True, max_length=MAX_LEN,
        padding=True, return_tensors="pt",
    )
    # (5, seq_len) -> (1, 5, seq_len): the model expects a batch dimension
    enc = {k: v.unsqueeze(0) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits

    return torch.softmax(logits, dim=1)[0].tolist()


# ---------------------------------------------------------------- ui
st.title("Smart MCQ Solver")
st.caption(
    "DeBERTa-v3 fine-tuned with LoRA. Enter a question and five options; "
    "the model ranks the three most likely answers."
)

EXAMPLE = {
    "q": "What is Martin Heidegger's view on the relationship between humans and time?",
    "o": [
        "Heidegger believes humans exist within an infinite time continuum.",
        "Heidegger believes humans are time itself, not merely within it.",
        "Heidegger does not believe in the existence of time at all.",
        "Heidegger sees the relationship as fundamentally cyclical.",
        "Heidegger argues that time is an illusion of consciousness.",
    ],
}

if "loaded_example" not in st.session_state:
    st.session_state.loaded_example = False

if st.button("Load an example"):
    st.session_state.loaded_example = True

d = EXAMPLE if st.session_state.loaded_example else {"q": "", "o": [""] * 5}

question = st.text_area("Question", value=d["q"], height=80,
                        placeholder="Type the question here")

st.markdown("**Options**")
options = [
    st.text_input(f"Option {L}", value=d["o"][i], key=f"opt_{L}")
    for i, L in enumerate(OPTS)
]

# ---------------------------------------------------------------- run
if st.button("Solve", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    elif any(not o.strip() for o in options):
        st.warning("Please fill in all five options.")
    else:
        tokenizer, model = load_model()

        with st.spinner("Thinking..."):
            probs = predict(question.strip(), [o.strip() for o in options],
                            tokenizer, model)

        ranked = sorted(range(5), key=lambda i: probs[i], reverse=True)
        top3 = ranked[:3]

        st.success(f"Prediction: **{' '.join(OPTS[i] for i in top3)}**")

        st.markdown("**Ranked answers**")
        for rank, i in enumerate(top3, start=1):
            st.markdown(f"**{rank}. Option {OPTS[i]}** — confidence {probs[i]:.1%}")
            st.markdown(
                f"<div style='margin:-8px 0 10px 18px;color:#555'>{options[i]}</div>",
                unsafe_allow_html=True,
            )

        st.markdown("**All five scores**")
        st.bar_chart({OPTS[i]: probs[i] for i in range(5)})

        conf = max(probs)
        if conf < 0.5:
            st.info(
                f"Top confidence is only {conf:.1%}. On the validation set the "
                "model averaged 0.94 when correct and 0.36 when wrong, so a low "
                "score here means it is genuinely unsure."
            )

# ---------------------------------------------------------------- about
with st.expander("About this project"):
    st.markdown(
        """
**Task.** Each question has five options and exactly one is correct.
The model ranks them and returns the top three, scored by MAP@3.

**Final model.** A blend of DeBERTa-v3 fine-tuned with LoRA and an ELECTRA
model, scoring **0.75644** on the public leaderboard against a 0.73 cutoff
and a 0.3578 random baseline. This demo runs the DeBERTa half only, since
loading two transformers exceeds the memory available here.

**What LoRA does.** It freezes the pretrained weights and trains small
low-rank adapters instead — 1,476,097 parameters out of 185,899,010, which
is 0.79% — while still reaching 0.9983 validation MAP@3.

**A caveat worth stating.** The competition dataset contains an annotation
artifact: correct answers are written in a subtly different style from the
distractors, so a model that never sees the question still scores 100% on
validation. My model agrees with such a question-blind detector on 99.7% of
rows, so its performance on this dataset does not fully transfer to
questions written differently.
        """
    )

st.caption("Sagar K Chaudhary| Deep Learning Project")
