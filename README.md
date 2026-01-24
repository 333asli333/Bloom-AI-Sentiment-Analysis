# 🌸 Bloom AI: Generative Text Application

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b)
![Generative AI](https://img.shields.io/badge/AI-Generative-purple)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Project Overview
**Bloom AI** is an interactive Generative AI application designed to explore the capabilities of Large Language Models (LLMs). Built with **Streamlit**, it serves as a bridge between complex AI architectures and a user-friendly web interface, allowing users to generate creative text, stories, and answers in real-time.

This project demonstrates the integration of **Natural Language Processing (NLP)** models into a responsive web environment.

## 🎥 Project Demo
*(Watch the demo video above to see the application in action)*

## 🚀 Key Features
* **Interactive UI:** A clean, modern interface built with Streamlit (`app.py`).
* **Generative Capabilities:** Utilizes advanced Transformer architectures to generate coherent text based on user prompts.
* **Real-Time Inference:** Fast response generation with visual feedback.
* **Customizable Parameters:** (Optional) Users can tweak generation settings like temperature and max length.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Frontend Framework:** Streamlit
* **Core Libraries:** TensorFlow / PyTorch, Transformers, NumPy
* **Model:** Bloom / Custom LLM Architecture

## ⚙️ Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/aslitorun/bloom-ai.git](https://github.com/aslitorun/bloom-ai.git)
    cd bloom-ai
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Ensure `streamlit` and `tensorflow` are included in your requirements.txt)*

3.  **Run the application:**
    ```bash
    streamlit run app.py
    ```

## 🧠 How It Works
The application takes a user's text prompt, processes it through the tokenizer, and feeds it into the pre-trained Transformer model. The model predicts the most probable next tokens to construct a meaningful response, which is then streamed back to the user interface.

## 👩‍💻 Author
**Aslı Torun**
*Data Scientist & AI Engineer*
[LinkedIn](https://www.linkedin.com/in/aslitorun/) | [GitHub](https://github.com/aslitorun)
