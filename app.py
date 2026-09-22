import streamlit as st
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from PIL import Image
import cv2

# Page config
st.set_page_config(
    page_title="Brain Tumor Segmentation",
    page_icon="🧠",
    layout="wide"
)

# Custom functions
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = K.flatten(tf.cast(y_pred, tf.float32))
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def iou(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = K.flatten(tf.cast(y_pred, tf.float32))
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        'models/best_model.keras',
        custom_objects={
            'bce_dice_loss': bce_dice_loss,
            'dice_coef'    : dice_coef,
            'iou'          : iou
        }
    )
    return model

def preprocess_image(image):
    img = np.array(image)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.resize(img, (256, 256))
    img = img / 255.0
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return img

def predict(model, img):
    with tf.device('/CPU:0'):
        pred = model.predict(img, verbose=0)
    pred_binary = (pred > 0.5).astype(np.float32)
    return pred[0].squeeze(), pred_binary[0].squeeze()

def create_overlay(original, mask, color='red'):
    img = np.array(original.resize((256, 256))) / 255.0
    if len(img.shape) == 2:
        img = np.stack([img]*3, axis=-1)
    overlay = img.copy()
    if color == 'red':
        overlay[mask > 0.5, 0] = 1.0
        overlay[mask > 0.5, 1] = 0.0
        overlay[mask > 0.5, 2] = 0.0
    return (overlay * 255).astype(np.uint8)

# ============================================================
# APP UI
# ============================================================

st.title("🧠 Brain Tumor Segmentation")
st.markdown("### Automated tumor detection using U-Net Deep Learning")
st.markdown("---")

# Sidebar
st.sidebar.title("About")
st.sidebar.markdown("""
**Model:** U-Net CNN  
**Dataset:** BRISC 2025  
**Test Dice Score:** 0.8280  
**Test Accuracy:** 98.96%  
""")
st.sidebar.markdown("---")
st.sidebar.markdown("Upload a brain MRI scan to get automated tumor segmentation!")

# Main area
st.markdown("## Upload MRI Scan")
uploaded_file = st.file_uploader(
    "Choose a brain MRI image",
    type=['jpg', 'jpeg', 'png'],
    help="Upload a T1-weighted brain MRI scan"
)

if uploaded_file is not None:
    # Load and display original
    image = Image.open(uploaded_file).convert('RGB')

    st.markdown("---")
    st.markdown("## Results")

    # Load model
    with st.spinner('Loading model...'):
        model = load_model()

    # Preprocess and predict
    with st.spinner('Analyzing MRI scan...'):
        img_processed = preprocess_image(image)
        pred_mask, pred_binary = predict(model, img_processed)

    # Check if tumor detected
    tumor_pixels = np.sum(pred_binary > 0.5)
    total_pixels = pred_binary.shape[0] * pred_binary.shape[1]
    tumor_percentage = (tumor_pixels / total_pixels) * 100
    tumor_detected = tumor_pixels > 100

    # Show detection result
    if tumor_detected:
        st.error(f"⚠️ TUMOR DETECTED — {tumor_percentage:.2f}% of scan area")
    else:
        st.success("✅ NO TUMOR DETECTED")

    st.markdown("---")

    # Show images in 3 columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Original MRI**")
        st.image(image.resize((256, 256)), use_column_width=True)

    with col2:
        st.markdown("**Predicted Tumor Mask**")
        mask_display = (pred_binary * 255).astype(np.uint8)
        st.image(mask_display, use_column_width=True)

    with col3:
        st.markdown("**Tumor Overlay (Red)**")
        overlay = create_overlay(image, pred_binary)
        st.image(overlay, use_column_width=True)

    st.markdown("---")

    # Stats
    st.markdown("## Prediction Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Tumor Detected", "YES" if tumor_detected else "NO")
    with col2:
        st.metric("Tumor Area", f"{tumor_percentage:.2f}%")
    with col3:
        st.metric("Model Confidence", f"{float(pred_mask.max()):.2%}")

else:
    # Show instructions when no image uploaded
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("📤 Step 1: Upload MRI scan above")
    with col2:
        st.info("🤖 Step 2: Model analyzes the scan")
    with col3:
        st.info("🎯 Step 3: See tumor segmentation results")