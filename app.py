import streamlit as st
import joblib
from pathlib import Path

st.set_page_config(
    page_title="SMS Spam Detection",
    page_icon="📩",
    layout="centered",
)

MODEL_PATH = Path(__file__).parent / "model.joblib"

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

model = load_model()

st.title("📩 SMS Spam Detection")
st.write(
    "Enter an SMS message and let the trained machine-learning model classify it "
    "as **Spam** or **Ham (Not Spam)**."
)

if model is None:
    st.error(
        "The trained model file `model.joblib` is missing. "
        "Export your trained pipeline to this file and place it beside `app.py`."
    )
    st.code("import joblib\njoblib.dump(model, 'model.joblib')")
    st.stop()

examples = {
    "Promotional example": "Congratulations! You have won a free prize. Call now to claim.",
    "Normal example": "Hey, are we meeting tomorrow?",
}
chosen = st.selectbox("Try an example (optional)", ["— Choose an example —", *examples.keys()])
default_text = examples.get(chosen, "")

message = st.text_area(
    "SMS message",
    value=default_text,
    placeholder="Paste or type your SMS message here…",
    height=160,
    max_chars=5000,
)

if st.button("Check Message", type="primary", use_container_width=True):
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        prediction = str(model.predict([message])[0]).strip().lower()
        if prediction == "spam":
            st.error("🚨 Prediction: SPAM")
            st.write("The model classifies this message as spam.")
        else:
            st.success("✅ Prediction: HAM (NOT SPAM)")
            st.write("The model classifies this message as a normal, non-spam message.")

        # This is the model's probability estimate, not a guaranteed certainty.
        if hasattr(model, "predict_proba"):
            try:
                probabilities = model.predict_proba([message])[0]
                classes = list(model.classes_)
                if prediction in classes:
                    score = float(probabilities[classes.index(prediction)])
                    st.caption(f"Model probability estimate for the predicted class: {score:.1%}")
            except Exception:
                pass

st.divider()
with st.expander("About this app"):
    st.write(
        "This app uses the trained SMS spam classification pipeline. "
        "It predicts only the classes represented in the training data (usually spam and ham). "
        "A prediction can be wrong, and the probability estimate should not be treated as a guarantee."
    )
