import streamlit as st
import requests
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import numpy as np
import cv2
from PIL import Image
import os

# Dropbox model link
MODEL_URL = "https://www.dropbox.com/scl/fi/m8e7tr4vy887rrmedvpok/model_final-1.pth?rlkey=bf5ov8r1m89u9qp88alpuvmse&st=htkj8ux1&dl=1"
MODEL_PATH = "model_final.pth"

# Demo image URL
DEMO_IMAGE_URL = "https://raw.githubusercontent.com/sunnycodes3114/trial-3/refs/heads/main/gettyimages-157561077-1024x1024.jpg"

# Download model if not exists
if not os.path.exists(MODEL_PATH):
    st.write("Downloading model...")
    response = requests.get(MODEL_URL, stream=True)
    with open(MODEL_PATH, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    st.write("Model downloaded.")

# Configure Detectron2
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8
cfg.MODEL.WEIGHTS = MODEL_PATH
cfg.MODEL.DEVICE = "cpu"  # Hugging Face Spaces free tier doesn't support GPU

predictor = DefaultPredictor(cfg)

# Metadata
MetadataCatalog.get("car_parts").set(thing_classes=[
    "Dent", "Scratch", "Broken part", "Paint chip", 
    "Missing part", "Flaking", "Corrosion", "Cracked"
])
metadata = MetadataCatalog.get("car_parts")

# Streamlit app
st.title("Car Parts Damage Detection")

# Option to use demo image or upload a custom image
use_demo_image = st.checkbox("Use Demo Image")

if use_demo_image:
    # Load the demo image from the URL
    response = requests.get(DEMO_IMAGE_URL)
    demo_image = Image.open(requests.get(DEMO_IMAGE_URL, stream=True).raw)
    st.image(demo_image, caption="Demo Image", use_column_width=True)
    image_to_process = demo_image
else:
    # Option to upload a custom image
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png"])
    if uploaded_file:
        image_to_process = Image.open(uploaded_file)
        st.image(image_to_process, caption="Uploaded Image", use_column_width=True)

# Process the selected image if available
if use_demo_image or (not use_demo_image and uploaded_file):
    # Convert image for processing
    image_cv2 = cv2.cvtColor(np.array(image_to_process), cv2.COLOR_RGB2BGR)
    
    # Run predictions
    outputs = predictor(image_cv2)

    # Visualize predictions
    v = Visualizer(image_cv2[:, :, ::-1], metadata=metadata, scale=1.0)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    visualized_image = out.get_image()[:, :, ::-1]

    # Display the overlay
    st.image(visualized_image, caption="Detected Damage Overlay", use_column_width=True)
