# 🤖 Advanced English FAQ Chatbot

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Sentence-BERT](https://img.shields.io/badge/Sentence--BERT-FF9900?style=for-the-badge&logo=huggingface&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

An intelligent **English-only FAQ Chatbot** that uses **Sentence-BERT (SBERT)** for semantic search and Natural Language Processing (NLP) to understand user intent and sentiment.

Unlike simple keyword matching, this bot understands the *meaning* behind your questions. It features a Jupyter Notebook for experimentation and a production-ready Streamlit application.

---

## 🌟 Key Features

- **🧠 Semantic Search:** Powered by `all-MiniLM-L6-v2` to retrieve answers based on meaning.
- **📊 Context-Aware Analysis:** Detects user intent (e.g., *Tracking*, *Returns*, *Shipping*) and sentiment (*Positive*, *Negative*, *Neutral*).
- **🎭 Adaptive Tone:** Adjusts the response tone (Empathetic, Informative, Friendly) based on the user's emotion.
- **📂 Dual Data Source:** Capable of loading datasets from Hugging Face or using a robust local fallback for offline demos.
- **🚀 Interactive UI:** A clean, chat-based interface built with **Streamlit**.

---

## 📂 Project Structure

| File/Folder | Description |
| :--- | :--- |
| `app.py` | 🟢 **Main Application:** The Streamlit app source code. |
| `CodeAlpha_Chatbot_for_FAQs.ipynb` | 📓 **Jupyter Notebook:** Step-by-step NLP pipeline for testing and debugging. |
| `requirements.txt` | 📦 **Dependencies:** List of required Python libraries. |
| `Demo/` | 🎥 **Demo Assets:** Contains screenshots and video demonstrations of the project. |
| `LICENSE` | ⚖️ **MIT License:** Usage rights. |
| `README.md` | 📄 **Documentation:** This file. |

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/YourUsername/Advanced-FAQ-Chatbot.git](https://github.com/YourUsername/Advanced-FAQ-Chatbot.git)
cd Advanced-FAQ-Chatbot

```

### 2. Install Dependencies

Ensure you have Python installed, then run:

```bash
pip install -r requirements.txt

```

### 3. Run the Application

To start the Streamlit interface locally:

```bash
streamlit run app.py

```

*The app will open automatically in your browser at `http://localhost:8501*`

---

## 💻 How to Use

1. **Launch the App:** Follow the installation steps above.
2. **Type a Question:** In the chat input, ask questions like:
* *"Where is my order?"*
* *"I want to return a damaged item."*
* *"How do I reset my password?"*


3. **View Analysis:** Expand the **"See Analysis Details"** section in the chat to view the detected intent, sentiment, confidence score, and matched FAQ.

---

## 📸 Demo

Check out the `Demo` folder for a video walkthrough and screenshots of the chatbot in action.

---

## 🤖 Model Details

* **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`
* **NLP Tools:** NLTK (Tokenization, Stopwords), Scikit-Learn (Cosine Similarity).
* **Framework:** Streamlit (Frontend).

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the **MIT License**. See the [LICENSE](https://www.google.com/search?q=LICENSE) file for more details.

---

<p align="center">
<strong>Developed by Eslam Alsaeed</strong>
</p>

```

```
