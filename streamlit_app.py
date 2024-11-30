import os
import subprocess
try:
    import torch
except ImportError:
    subprocess.run(["pip", "install", "torch==2.0.1+cpu", "torchvision==0.15.2+cpu", "torchaudio==2.0.2+cpu"])

import streamlit as st

st.title("🎈 My new apfgrfgp")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)
