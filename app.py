"""Streamlit dashboard for the Fake Review Detection System."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from src.prediction import PredictionResult, predict_dataframe, predict_review, supports_explainability
from src.utils import load_artifacts, load_metrics

st.set_page_config(
    page_title="Fake Review Detection System",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    """Apply the small visual design system used across the dashboard."""

    st.markdown(
        """
        <style>
        :root { --navy: #071426; --navy-2: #0d1f38; --purple: #8557ff;
                --cyan: #39d8ff; --text: #eaf4ff; --muted: #a9bad1; --card: #102440; }
        .stApp { background: radial-gradient(circle at 80% 0%, #1d1a4a 0%, transparent 32%),
                           linear-gradient(140deg, var(--navy) 0%, #0b1930 56%, #08101f 100%); color: var(--text); }
        [data-testid="stSidebar"] { background: rgba(6, 17, 33, .92); border-right: 1px solid rgba(123, 96, 255, .24); }
        [data-testid="stSidebar"] * { color: var(--text); }
        h1, h2, h3 { color: var(--text) !important; letter-spacing: -0.025em; }
        p, li { color: var(--muted); }
        .hero { padding: 2.6rem 2.4rem; border: 1px solid rgba(81, 215, 255, .22); border-radius: 24px;
                background: linear-gradient(120deg, rgba(33, 52, 100, .78), rgba(102, 51, 176, .27));
                box-shadow: 0 18px 50px rgba(0, 0, 0, .22); margin-bottom: 1.4rem; }
        .eyebrow { color: var(--cyan); font-size: .78rem; font-weight: 700; text-transform: uppercase; letter-spacing: .14em; }
        .hero h1 { font-size: clamp(2rem, 4vw, 3.45rem); margin: .35rem 0 .65rem; }
        .hero p { max-width: 760px; font-size: 1.08rem; line-height: 1.65; }
        .card { height: 100%; padding: 1.35rem; border-radius: 18px; border: 1px solid rgba(168, 187, 220, .16);
                background: linear-gradient(150deg, rgba(20, 47, 78, .9), rgba(16, 31, 57, .94));
                transition: transform .2s ease, border-color .2s ease; }
        .card:hover { transform: translateY(-3px); border-color: rgba(57, 216, 255, .55); }
        .card h3 { margin: .25rem 0 .45rem; font-size: 1.08rem; }
        .icon { font-size: 1.65rem; }
        .flow { padding: 1rem 1.25rem; text-align: center; border-radius: 14px; font-weight: 700; color: var(--text);
                background: rgba(122, 87, 255, .15); border: 1px solid rgba(133, 87, 255, .38); }
        .arrow { color: var(--cyan); font-size: 1.6rem; text-align: center; padding-top: .6rem; }
        .disclaimer { padding: 1rem 1.2rem; border-left: 3px solid var(--cyan); border-radius: 0 12px 12px 0;
                background: rgba(57, 216, 255, .08); color: #c9edff; margin-top: 1.5rem; }
        .result { padding: 1.8rem; border-radius: 22px; border: 1px solid rgba(168, 187, 220, .22);
                  background: linear-gradient(135deg, rgba(16, 46, 72, .95), rgba(32, 24, 71, .95)); }
        .result-label { font-size: 1.75rem; margin: .15rem 0 .5rem; font-weight: 800; }
        .risk-high { color: #ff8d9c; } .risk-low { color: #65ecc5; }
        .confidence { font-size: 2.6rem; line-height: 1; color: var(--cyan); font-weight: 800; }
        .feature { display: inline-block; padding: .35rem .65rem; margin: .2rem .28rem .1rem 0; border-radius: 999px;
                   background: rgba(57, 216, 255, .12); color: #c5f6ff; border: 1px solid rgba(57, 216, 255, .28); }
        .stButton > button, .stDownloadButton > button { width: 100%; border: 0; border-radius: 10px; padding: .62rem 1rem;
                   font-weight: 700; color: #061120; background: linear-gradient(100deg, var(--cyan), #9bb4ff); transition: .2s; }
        .stButton > button:hover, .stDownloadButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 22px rgba(57, 216, 255, .25); }
        [data-testid="stFileUploader"] { border: 1px dashed rgba(57, 216, 255, .55); border-radius: 14px; padding: .4rem; }
        [data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
        .stTabs [data-baseweb="tab-list"] { gap: .3rem; } .stTabs [data-baseweb="tab"] { border-radius: 9px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def get_model_bundle() -> tuple[Any, Any] | None:
    """Cache disk-loaded artifacts for the lifetime of a Streamlit session."""

    return load_artifacts()


def render_hero(title: str, description: str, eyebrow: str = "ML Review Intelligence") -> None:
    """Show a consistent page introduction."""

    st.markdown(
        f"""<section class="hero"><div class="eyebrow">{eyebrow}</div><h1>{title}</h1>
        <p>{description}</p></section>""",
        unsafe_allow_html=True,
    )


def render_model_status(bundle: tuple[Any, Any] | None) -> None:
    """Explain the next action when model artifacts are not available yet."""

    if bundle is None:
        st.warning(
            "No trained model was found. Run `python train_model.py --data data/sample_reviews.csv` "
            "in this project folder, then refresh the app."
        )


def render_home(bundle: tuple[Any, Any] | None) -> None:
    """Render the product overview page."""

    render_hero(
        "Fake Review Detection System",
        "A practical machine-learning screening tool that looks for patterns in review wording. "
        "Use it to prioritise human review—not to make final accusations.",
    )
    columns = st.columns(3, gap="large")
    cards = [
        ("✦", "AI-based analysis", "TF-IDF text signals and trained classifiers assess review language."),
        ("◌", "Confidence scoring", "See how strongly the chosen model leans toward its assessment."),
        ("▦", "Bulk CSV analysis", "Screen a dataset, explore result patterns, and export the enriched file."),
    ]
    for column, (icon, heading, body) in zip(columns, cards):
        with column:
            st.markdown(
                f'<div class="card"><div class="icon">{icon}</div><h3>{heading}</h3><p>{body}</p></div>',
                unsafe_allow_html=True,
            )

    st.markdown("### How it works")
    flow = st.columns([4, 1, 4, 1, 4])
    flow[0].markdown('<div class="flow">✎<br>Review Text</div>', unsafe_allow_html=True)
    flow[1].markdown('<div class="arrow">→</div>', unsafe_allow_html=True)
    flow[2].markdown('<div class="flow">⚙<br>ML Analysis</div>', unsafe_allow_html=True)
    flow[3].markdown('<div class="arrow">→</div>', unsafe_allow_html=True)
    flow[4].markdown('<div class="flow">◉<br>Risk Result</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="disclaimer"><strong>Important:</strong> This is a screening tool, not definitive proof of fraud. '
        'Review context, reviewer history, and human judgement still matter.</div>',
        unsafe_allow_html=True,
    )
    render_model_status(bundle)


def render_result(result: PredictionResult, rating: int | None, explainable: bool) -> None:
    """Render one polished prediction card."""

    is_suspicious = result.label == "Suspicious Review"
    colour_class = "risk-high" if is_suspicious else "risk-low"
    st.markdown('<div class="result">', unsafe_allow_html=True)
    st.markdown("<div class='eyebrow'>Screening assessment</div>", unsafe_allow_html=True)
    st.markdown(f'<div class="result-label {colour_class}">{result.label}</div>', unsafe_allow_html=True)
    left, right = st.columns([1, 2])
    with left:
        st.markdown(f'<div class="confidence">{result.confidence:.0%}</div>', unsafe_allow_html=True)
        st.caption("model confidence")
    with right:
        st.markdown(f"**Risk indicator:** {result.risk_category}")
        st.write(result.explanation)
        if rating is not None:
            st.caption(f"Optional rating recorded: {rating}/5. The current classifier evaluates review text only.")
        if result.was_truncated:
            st.caption("For responsiveness, only the first 10,000 characters were analysed.")

    if result.influential_features:
        st.markdown("**Influential wording for this assessment**")
        chips = "".join(
            f'<span class="feature">{feature.feature} · {feature.contribution:.2f}</span>'
            for feature in result.influential_features
        )
        st.markdown(chips, unsafe_allow_html=True)
    elif not explainable:
        st.caption("Feature-level explanations are available when the saved model is Logistic Regression.")
    st.markdown("</div>", unsafe_allow_html=True)


def render_single_review(bundle: tuple[Any, Any] | None) -> None:
    """Render the interactive individual-review assessor."""

    render_hero(
        "Check One Review",
        "Paste a review to receive a cautious, model-based assessment. Empty and unusually long input are handled safely.",
        "Individual screening",
    )
    render_model_status(bundle)
    review = st.text_area(
        "Review text", height=220, placeholder="Paste a customer review here…", key="review_input"
    )
    include_rating = st.checkbox("Include the review's star rating (optional)")
    rating = st.slider("Star rating", 1, 5, 5) if include_rating else None
    if st.button("Analyze Review", type="primary", disabled=bundle is None):
        try:
            model, vectorizer = bundle  # type: ignore[misc]
            st.session_state["single_result"] = predict_review(model, vectorizer, review)
            st.session_state["single_rating"] = rating
        except ValueError as error:
            st.error(str(error))

    result = st.session_state.get("single_result")
    if result:
        model = bundle[0] if bundle else None
        render_result(result, st.session_state.get("single_rating"), bool(model and supports_explainability(model)))


def _make_chart_layout(figure: Any) -> Any:
    """Apply dashboard colours to a Plotly figure."""

    return figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#eaf4ff"},
        margin={"l": 20, "r": 20, "t": 45, "b": 20},
    )


def render_bulk_analysis(bundle: tuple[Any, Any] | None) -> None:
    """Render CSV upload, validation, prediction, plots, and export."""

    render_hero(
        "Bulk CSV Analysis",
        "Upload reviews, confirm the preview, then enrich each usable row with a cautious model assessment.",
        "Batch screening",
    )
    render_model_status(bundle)
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is None:
        st.info("Your file must include a `review_text` column. The `rating` column is optional.")
        return

    try:
        uploaded_data = pd.read_csv(uploaded_file)
        if "review_text" not in uploaded_data.columns:
            st.error("This CSV is missing the required `review_text` column.")
            return
        st.success(f"CSV ready: {len(uploaded_data):,} rows detected.")
        st.markdown("#### Preview")
        st.dataframe(uploaded_data.head(10), use_container_width=True, hide_index=True)
    except (UnicodeDecodeError, pd.errors.ParserError) as error:
        st.error(f"We could not read that CSV: {error}")
        return

    if st.button("Process CSV", type="primary", disabled=bundle is None):
        with st.spinner("Analysing review language…"):
            model, vectorizer = bundle  # type: ignore[misc]
            st.session_state["bulk_results"] = predict_dataframe(model, vectorizer, uploaded_data)

    results = st.session_state.get("bulk_results")
    if isinstance(results, pd.DataFrame):
        st.markdown("#### Processed results")
        st.dataframe(results, use_container_width=True, hide_index=True)
        usable_results = results.loc[results["predicted_label"].ne("Needs Review")]
        if not usable_results.empty:
            chart_one, chart_two = st.columns(2, gap="large")
            with chart_one:
                pie = px.pie(
                    usable_results,
                    names="predicted_label",
                    title="Assessment mix",
                    color="predicted_label",
                    color_discrete_map={"Likely Genuine": "#65ecc5", "Suspicious Review": "#ff8d9c"},
                )
                st.plotly_chart(_make_chart_layout(pie), use_container_width=True)
            with chart_two:
                counts = usable_results["risk_category"].value_counts().rename_axis("Risk category").reset_index(name="Reviews")
                bar = px.bar(
                    counts, x="Risk category", y="Reviews", title="Risk categories",
                    color="Risk category", color_discrete_sequence=["#8557ff", "#39d8ff", "#ff8d9c", "#65ecc5"],
                )
                st.plotly_chart(_make_chart_layout(bar), use_container_width=True)

        csv_data = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download processed CSV", data=csv_data, file_name="review_screening_results.csv", mime="text/csv"
        )


def _comparison_table(metrics: dict[str, Any]) -> pd.DataFrame:
    """Turn saved model metrics into a tidy table for app and learners."""

    comparison = pd.DataFrame(metrics.get("model_comparison", {})).T
    if comparison.empty:
        return comparison
    return comparison.drop(columns=["confusion_matrix"], errors="ignore").rename(
        columns={
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall",
            "macro_f1": "Macro F1",
        }
    )


def render_model_insights(bundle: tuple[Any, Any] | None) -> None:
    """Explain the model workflow and show persisted metrics when available."""

    render_hero(
        "Model Insights",
        "A transparent overview of the input format, text features, model comparison, and the limits of automated screening.",
        "Learning centre",
    )
    tabs = st.tabs(["Dataset & preprocessing", "Model comparison", "Metrics & limitations"])
    with tabs[0]:
        st.markdown("### Dataset format")
        st.code("review_text,label,rating\n\"Quick delivery and exactly as described\",genuine,5", language="csv")
        st.write(
            "`review_text` and `label` are required for training. Labels accept `fake`, `genuine`, `1`, or `0`; "
            "`rating` is optional metadata. Empty reviews are skipped during training and marked for review in bulk mode."
        )
        st.markdown("### Preprocessing")
        st.write(
            "Text is lowercased, URLs and unneeded punctuation are removed, whitespace is normalised, and NLTK tokenisation is used. "
            "Negation words such as `not` are intentionally preserved. Reviews are capped at 10,000 characters."
        )
        st.markdown("### Features")
        st.write(
            "TF-IDF combines word 1–2 grams with character 3–5 grams. This captures both meaningful phrases and small spelling/style patterns."
        )
    with tabs[1]:
        st.markdown("### Candidate models")
        st.write(
            "Training compares Multinomial Naive Bayes, Logistic Regression, and a calibrated Linear SVM. "
            "The highest macro F1 model is saved; Logistic Regression wins exact ties and is the default baseline."
        )
        metrics = load_metrics()
        if metrics:
            st.success(f"Saved model: {metrics.get('best_model', 'Unknown')} · selected by macro F1")
            table = _comparison_table(metrics)
            st.dataframe(table.style.format("{:.2%}"), use_container_width=True)
            st.caption(f"Evaluation holdout: {metrics.get('test_rows', '—')} rows from {metrics.get('training_rows', '—')} usable rows.")
        else:
            st.info("Train a model to populate the comparison table and evaluation results.")
    with tabs[2]:
        st.markdown("### What the metrics mean")
        st.write(
            "Accuracy is overall correctness. Precision asks how often suspicious flags are right. Recall asks how many suspicious examples were found. "
            "Macro F1 balances performance across genuine and suspicious classes, so it is used for model selection."
        )
        metrics = load_metrics()
        if metrics:
            best = metrics.get("best_model")
            matrix = metrics.get("model_comparison", {}).get(best, {}).get("confusion_matrix")
            if matrix:
                matrix_data = pd.DataFrame(matrix, index=["Actual genuine", "Actual suspicious"], columns=["Predicted genuine", "Predicted suspicious"])
                st.markdown("#### Saved model confusion matrix")
                st.dataframe(matrix_data, use_container_width=True)
        st.warning(
            "Limitations: language patterns are not evidence of fraud. Results can reflect biased or unrepresentative training data, "
            "may not transfer to new products or languages, and should never be the sole basis for penalties or moderation decisions."
        )
    render_model_status(bundle)


def main() -> None:
    """Start the dashboard."""

    inject_styles()
    bundle = get_model_bundle()
    with st.sidebar:
        st.markdown("## 🔎 Review Guard")
        st.caption("Machine-learning screening dashboard")
        page = st.radio(
            "Navigate",
            ["Home / Dashboard", "Check One Review", "Bulk CSV Analysis", "Model Insights"],
            label_visibility="collapsed",
        )
        st.divider()
        if bundle is not None:
            metrics = load_metrics() or {}
            st.success("Model ready")
            st.caption(f"Active model: {metrics.get('best_model', 'Saved classifier')}")
        else:
            st.warning("Training needed")
        st.caption("A decision-support tool for responsible review screening.")

    if page == "Home / Dashboard":
        render_home(bundle)
    elif page == "Check One Review":
        render_single_review(bundle)
    elif page == "Bulk CSV Analysis":
        render_bulk_analysis(bundle)
    else:
        render_model_insights(bundle)


if __name__ == "__main__":
    main()
