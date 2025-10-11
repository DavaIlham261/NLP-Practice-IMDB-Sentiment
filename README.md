# Sentiment Analysis of IMDB Movie Reviews with Transformer

![GitHub last commit](https://img.shields.io/github/last-commit/DavaIlham261/NLP-Practice-IMDB-Sentiment)
[![Hugging Face](https://img.shields.io/badge/Model%20on%20Hub-Hugging%20Face-yellow)](https://huggingface.co/DavaIlham/distilbert-finetuned-sentiment-imdb)

This project explores the capabilities of modern AI models to understand emotional nuances in human language. A Transformer model **(DistilBERT)** was fine-tuned to perform binary sentiment classification (positive/negative) on the popular IMDB movie review dataset.

## ⚡️ How to Use the Model

The trained model can be used directly via the Hugging Face `transformers` library.

1. **Install the library:**
    ```bash
    pip install transformers
    ```
2. **Run the Python code:**
    ```python
    from transformers import pipeline

    # Load the model directly from Hugging Face Hub
    sentiment_analyzer = pipeline("sentiment-analysis", model="DavaIlham/distilbert-finetuned-sentiment-imdb")

    # Test with your own sentences
    results = sentiment_analyzer([
        "This movie was a masterpiece, the best I've seen all year!",
        "The plot was incredibly boring and predictable."
    ])

    print(results)
    ```

**➡️ View & try the model interactively on [Hugging Face Hub](https://huggingface.co/DavaIlham/distilbert-finetuned-sentiment-imdb)**

---

## 🌱 The Learning Journey

This project is the culmination of a structured self-learning journey, divided into several "Quests" to build AI Engineer skills from the ground up.

* **Quests 1-3 (Foundation):** Strengthening Python basics, mathematical intuition (Linear Algebra, Calculus, Statistics), and standard industry workflows using Git & GitHub.
* **Quest 4 (Data Analysis):** Mastering data analysis tools like NumPy and Pandas to dissect and clean datasets.
* **Quest 5 (Classical Machine Learning):** Building the first predictive model using Scikit-learn on the Titanic dataset, understanding the ML project lifecycle from preprocessing to evaluation.
* **Quest 6 (First Deep Learning):** Assembling the first artificial "brain" (Neural Network) using TensorFlow/Keras for image classification tasks on the Fashion MNIST dataset.
* **Quest 7 (NLP Specialization):** Applying Deep Learning to understand language, fine-tuning an advanced Transformer model (DistilBERT) using the Hugging Face ecosystem. This project is the result of Quest 7.

---

## 📈 Results & Performance

The model was evaluated on 25,000 reviews from the IMDB test set, which were never seen during training.

| Metric          | Score                        |
| --------------- | --------------------------- |
| **Accuracy** | `0.91188` |
| **Loss** | `0.45617571473121643`|
| **Precision** | `0.9120256263373727` |
| **Recall** | `0.91188` |
| **F1-Score** | `0.9118722130287433` |

*(**Note:** To fill this table, refer to the output from `trainer.evaluate()` and `classification_report()` in your notebook)*

---

## 🛠️ Tech Stack & Libraries

* **Python 3.x**
* **TensorFlow & Keras**: Backend and API for building Deep Learning models.
* **Hugging Face `transformers`**: For loading pre-trained models, tokenizers, and fine-tuning.
* **Hugging Face `datasets`**: For loading and processing the IMDB dataset.
* **Scikit-learn**: For evaluation metrics (`train_test_split`, `classification_report`).
* **Jupyter Notebook**: As the development environment.

---

## ⚙️ Reproducing Results

To rerun the training process locally, follow these steps:

1. **Clone this repository:**
    ```bash
    git clone [https://github.com/DavaIlham261/NLP-Practice-IMDB-Sentiment.git](https://github.com/DavaIlham261/NLP-Practice-IMDB-Sentiment.git)
    cd NLP-Practice-IMDB-Sentiment
    ```

2. **(Optional but recommended) Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    ```

3. **Install all dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Notebook:** Open and execute the cells in `quest-7.ipynb`.

---

## 🔮 Next Steps

Some ideas for future development of this project:
* Try fine-tuning larger Transformer models (such as RoBERTa or BERT-large).
* Perform multi-class sentiment analysis (e.g., very positive, positive, neutral, negative, very negative).
* Apply the model to review datasets in Indonesian.

---

## 👤 Author

* **Name:** Dava Ilham Muhammad
* **GitHub:** [@DavaIlham261](https://github.com/DavaIlham261)
* **LinkedIn:** [Dava Ilham Muhammad](https://www.linkedin.com/in/dava-muhammad-4861a3286/)
