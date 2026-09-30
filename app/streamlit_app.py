"""
Streamlit Application for Intelligent Plant Disease and Pest Detection System.

Features:
- Dual Detection Mode: CNN Plant Disease Classification & YOLOv8 Pest Detection.
- Upload image (.jpg, .jpeg, .png).
- Bounding-box visualization for YOLOv8 pest detection.
- Disease and pest information with preventive recommendations.
- Interactive metrics dashboard displaying actual evaluation results.
- Robust error handling for missing models/configs.
"""

import sys
import json
from pathlib import Path
from PIL import Image
import streamlit as st

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Page configuration
st.set_page_config(
    page_title="Plant Disease & Pest Vision AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E4D2B;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4A5568;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F7FAFC;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #2F855A;
        margin-bottom: 1rem;
    }
    .disclaimer {
        font-size: 0.85rem;
        color: #718096;
        background-color: #FEFCBF;
        padding: 0.75rem;
        border-radius: 6px;
        border: 1px solid #ECC94B;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)


def load_prediction_modules():
    """Import predictor modules safely."""
    disease_predictor = None
    pest_detector = None
    disease_err = None
    pest_err = None

    try:
        from src.disease_detection.predict_disease import DiseasePredictor
        cnn_model_path = PROJECT_ROOT / "models" / "cnn" / "best_disease_model.pth"
        if cnn_model_path.exists():
            disease_predictor = DiseasePredictor(model_path=str(cnn_model_path))
        else:
            disease_err = f"CNN model weights file not found at '{cnn_model_path}'. Please train the CNN model first."
    except Exception as e:
        disease_err = f"Could not load CNN predictor: {str(e)}"

    try:
        from src.pest_detection.predict_pest import PestDetector
        yolo_model_path = PROJECT_ROOT / "models" / "yolov8" / "pest_detector" / "weights" / "best.pt"
        if not yolo_model_path.exists():
            yolo_model_path = PROJECT_ROOT / "models" / "yolov8" / "best.pt"
        
        pest_detector = PestDetector(model_path=str(yolo_model_path))
    except Exception as e:
        pest_err = f"Could not load YOLOv8 detector: {str(e)}"

    return disease_predictor, pest_detector, disease_err, pest_err


def main():
    st.markdown("<div class='main-header'>🌿 Intelligent Plant Disease & Pest Detection System</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Deep Learning Vision System using Transfer Learning CNN & YOLOv8</div>", unsafe_allow_html=True)

    # Sidebar controls
    st.sidebar.header("🕹️ Control Panel")
    app_mode = st.sidebar.radio(
        "Select Feature Mode",
        ["🔍 Inference & Prediction", "📊 Model Evaluation & Metrics", "📁 Dataset Inspection Summary"]
    )

    if app_mode == "🔍 Inference & Prediction":
        run_inference_tab()
    elif app_mode == "📊 Model Evaluation & Metrics":
        run_metrics_tab()
    elif app_mode == "📁 Dataset Inspection Summary":
        run_inspection_tab()

    # Footer Disclaimer
    st.markdown(
        "<div class='disclaimer'>⚠️ <b>Notice:</b> Recommendations provided by this system are for educational "
        "and informational guidance only. Always consult with local agricultural extension officers or certified agronomists "
        "before applying chemical pesticides or treatments.</div>",
        unsafe_allow_html=True
    )


def run_inference_tab():
    st.subheader("📸 Upload Plant Image for AI Analysis")

    task_type = st.radio(
        "Select Detection Task:",
        ["🍃 Plant Disease Classification (CNN)", "🐛 Pest Detection & Localization (YOLOv8)"],
        horizontal=True
    )

    uploaded_file = st.file_uploader(
        "Choose a leaf or crop image...",
        type=["jpg", "jpeg", "png", "webp"]
    )

    disease_predictor, pest_detector, disease_err, pest_err = load_prediction_modules()

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)

        with col2:
            if "Disease" in task_type:
                st.markdown("### 🔬 Disease Classification Results")
                if disease_predictor is None:
                    st.error(disease_err or "CNN Model is not available.")
                else:
                    with st.spinner("Running CNN Disease Inference..."):
                        res = disease_predictor.predict(image)

                    st.success(f"**Predicted Condition:** {res['predicted_class']}")
                    st.metric(label="Confidence Score", value=f"{res['confidence_score'] * 100:.2f}%")

                    st.markdown("#### Top Probabilities")
                    for top in res['top_predictions']:
                        st.progress(top['confidence'], text=f"{top['class_name']}: {top['confidence']*100:.1f}%")

                    st.markdown("#### 📖 Description & Prevention")
                    st.write(res['description'])
                    st.markdown("**Management Recommendations:**")
                    for rec in res['recommendations']:
                        st.markdown(f"- {rec}")

            else:
                st.markdown("### 🎯 Pest Detection & Localization Results")
                if pest_detector is None:
                    st.warning(f"{pest_err}. Using fallback demo detection.")
                
                with st.spinner("Running YOLOv8 Pest Localization..."):
                    if pest_detector:
                        res, ann_img = pest_detector.detect(image)
                    else:
                        res, ann_img = {"pests_detected_count": 0, "detections": [], "description": "YOLOv8 model weights pending training.", "recommendations": []}, image

                st.image(ann_img, caption="YOLOv8 Detection Visualization", use_column_width=True)
                st.metric(label="Pests Detected Count", value=res['pests_detected_count'])

                if res['detections']:
                    st.markdown("#### Detection Details")
                    for det in res['detections']:
                        st.write(f"- **{det['class_name']}** (Confidence: {det['confidence']*100:.1f}%) | BBox: `{det['bbox']}`")

                st.markdown("#### 🛡️ Pest Control Recommendations")
                st.write(res['description'])
                for rec in res['recommendations']:
                    st.markdown(f"- {rec}")


def run_metrics_tab():
    st.subheader("📊 Actual Model Evaluation & Performance Report")

    tab1, tab2 = st.tabs(["CNN Disease Classification", "YOLOv8 Pest Detection"])

    with tab1:
        cnn_metrics_file = PROJECT_ROOT / "models" / "cnn" / "evaluation_metrics.json"
        cm_plot = PROJECT_ROOT / "models" / "cnn" / "confusion_matrix.png"
        lc_plot = PROJECT_ROOT / "models" / "cnn" / "learning_curves.png"

        if cnn_metrics_file.exists():
            with open(cnn_metrics_file, 'r') as f:
                metrics = json.load(f)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Overall Accuracy", f"{metrics.get('accuracy', 0)*100:.2f}%")
            c2.metric("Macro Precision", f"{metrics.get('macro_avg', {}).get('precision', 0):.4f}")
            c3.metric("Macro Recall", f"{metrics.get('macro_avg', {}).get('recall', 0):.4f}")
            c4.metric("Macro F1-Score", f"{metrics.get('macro_avg', {}).get('f1_score', 0):.4f}")

            if cm_plot.exists():
                st.image(str(cm_plot), caption="Test Confusion Matrix", use_column_width=True)
            if lc_plot.exists():
                st.image(str(lc_plot), caption="Training & Validation Learning Curves", use_column_width=True)
        else:
            st.info("CNN Evaluation Metrics are pending model training. Run `python src/disease_detection/evaluate_cnn.py` after training.")

    with tab2:
        yolo_metrics_file = PROJECT_ROOT / "models" / "yolov8" / "evaluation_metrics.json"
        if yolo_metrics_file.exists():
            with open(yolo_metrics_file, 'r') as f:
                ymetrics = json.load(f)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Precision", f"{ymetrics.get('precision', 0):.4f}")
            c2.metric("Recall", f"{ymetrics.get('recall', 0):.4f}")
            c3.metric("mAP@0.5", f"{ymetrics.get('mAP_50', 0):.4f}")
            c4.metric("mAP@0.5:0.95", f"{ymetrics.get('mAP_50_95', 0):.4f}")
        else:
            st.info("YOLOv8 Evaluation Metrics are pending model training. Run `python src/pest_detection/evaluate_yolov8.py` after training.")


def run_inspection_tab():
    st.subheader("📁 Dataset Inspection & Statistics")
    report_file = PROJECT_ROOT / "data" / "inspection_report.json"
    if report_file.exists():
        with open(report_file, 'r') as f:
            data = json.load(f)
        st.json(data)
    else:
        st.info("Dataset inspection report not generated yet. Run `python src/data/inspect_dataset.py`.")


if __name__ == "__main__":
    main()
