# AgriVision AI
A cozy AI-powered plant care assistant built with Streamlit and Teachable Machine.

||Plant Detector & Care Assistant||

A cute, minimalist web application that uses a dual-engine AI architecture to check if your green friends need a drink and provide full care diagnostics. Built with love using Python, Streamlit, Teachable Machine, and Google Gemini.
​> Features
​- AI Vision Classification: Uses a trained Keras model to instantly recognize if your plant is "Happy" or "Thirsty".
​- Full AI Diagnostics: Powered by Google's Gemini Vision API (gemini-2.5-flash) to generate structured care guides, including plant identification, disease treatment, watering intervals, and manure/fertilizer schedules.
​- Single-Asset Workflow: Take a photo with your camera or upload an image once, and run both instant classification and deep care diagnostics on demand.
​- Aesthetic & Cozy UI: Soft pastel colors and a stress-free layout designed to make plant care simple and enjoyable.
​- Cloud & Security Ready: Deployable on Streamlit Cloud with secure API key management via environment secrets.

Tech Stack
**Brain:** [Teachable Machine](https://teachablemachine.withgoogle.com/) (TensorFlow/Keras)
**Frontend:** [Streamlit](https://streamlit.io/)
**Editor:** VS Code
**Language:** Python 3.9+

Project Structure
* `app.py`: The main logic and UI code.
* `keras_model.h5`: The exported AI model.
* `labels.txt`: The classes (Happy vs. Thirsty).
* `requirements.txt`: Necessary libraries for the cloud to run the app.

How to use
1. Upload a photo of your plant.
2. Wait for the AI to analyze the leaves.
3. If it's thirsty, give it some 💧!
4. Deep Diagnostics: Click "Analyze Plant Health & Care" to unlock a full AI breakdown covering disease treatment, watering schedule, and fertilizer tips!

---
*Created with Vision by Khaloud ALtaf*
