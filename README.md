# Analisis Sentimen Ulasan Film IMDB dengan Transformer

![GitHub last commit](https://img.shields.io/github/last-commit/DavaIlham261/AI-Engineer-Practice)
[![Hugging Face](https://img.shields.io/badge/Model%20on%20Hub-Hugging%20Face-yellow)](https://huggingface.co/DavaIlham/distilbert-finetuned-sentiment-imdb)

Proyek ini mengeksplorasi kemampuan model AI modern untuk memahami nuansa emosi dalam bahasa manusia. Sebuah model Transformer **(DistilBERT)** di-*fine-tuning* untuk melakukan klasifikasi sentimen biner (positif/negatif) pada dataset ulasan film IMDB yang populer.



## ⚡️ Cara Menggunakan Model

Model yang sudah dilatih dapat langsung digunakan melalui *library* `transformers` dari Hugging Face.

1.  **Instal library:**
    ```bash
    pip install transformers
    ```
2.  **Jalankan kode Python:**
    ```python
    from transformers import pipeline

    # Muat model langsung dari Hugging Face Hub
    sentiment_analyzer = pipeline("sentiment-analysis", model="DavaIlham/distilbert-finetuned-sentiment-imdb")

    # Uji dengan kalimat Anda sendiri
    hasil = sentiment_analyzer([
        "This movie was a masterpiece, the best I've seen all year!",
        "The plot was incredibly boring and predictable."
    ])

    print(hasil)
    ```

**➡️ Lihat Model & coba secara interaktif di [Hugging Face Hub](https://huggingface.co/DavaIlham/distilbert-finetuned-sentiment-imdb)**

---

## 🌱 Perjalanan Belajar (The Learning Journey)

Proyek ini adalah puncak dari perjalanan belajar mandiri yang terstruktur, dibagi menjadi beberapa "Quest" untuk membangun skill sebagai AI Engineer dari dasar.

* **Quest 1-3 (Fondasi):** Memperkuat dasar-dasar Python, intuisi Matematika (Aljabar Linear, Kalkulus, Statistika), dan alur kerja standar industri menggunakan Git & GitHub.
* **Quest 4 (Analisis Data):** Menguasai perkakas analisis data seperti NumPy dan Pandas untuk membedah dan membersihkan dataset.
* **Quest 5 (Machine Learning Klasik):** Membangun model prediksi pertama menggunakan Scikit-learn pada dataset Titanic, memahami siklus hidup proyek ML dari pra-pemrosesan hingga evaluasi.
* **Quest 6 (Deep Learning Pertama):** Merakit "otak" buatan pertama (Neural Network) menggunakan TensorFlow/Keras untuk tugas klasifikasi gambar pada dataset Fashion MNIST.
* **Quest 7 (Spesialisasi NLP):** Menerapkan kekuatan Deep Learning untuk memahami bahasa, melakukan *fine-tuning* pada model Transformer (DistilBERT) canggih menggunakan ekosistem Hugging Face. Proyek ini adalah hasil dari Quest 7.

---

## 📈 Hasil & Performa

Model dievaluasi pada 25,000 ulasan dari *test set* IMDB yang belum pernah dilihat sebelumnya saat training.

| Metrik          | Skor                        |
| --------------- | --------------------------- |
| **Akurasi** | `0.91188` |
| **Loss** | `0.45617571473121643`|
| **Precision** | `0.9120256263373727` |
| **Recall** | `0.91188` |
| **F1-Score** | `0.9118722130287433` |

*(**Catatan:** Untuk mengisi tabel ini, lihat kembali output dari perintah `trainer.evaluate()` dan `classification_report()` di notebook Anda)*

---

## 🛠️ Tech Stack & Library

* **Python 3.x**
* **TensorFlow & Keras**: Sebagai backend dan API untuk membangun model Deep Learning.
* **Hugging Face `transformers`**: Untuk memuat model pre-trained, tokenizer, dan proses fine-tuning.
* **Hugging Face `datasets`**: Untuk memuat dan memproses dataset IMDB.
* **Scikit-learn**: Untuk metrik evaluasi (`train_test_split`, `classification_report`).
* **Jupyter Notebook**: Sebagai environment development.

---

## ⚙️ Reproduksi Hasil

Untuk menjalankan ulang proses training di lingkungan lokal Anda, ikuti langkah-langkah berikut:

1.  **Clone repositori ini:**
    ```bash
    git clone [https://github.com/DavaIlham261/AI-Engineer-Practice.git](https://github.com/DavaIlham261/AI-Engineer-Practice.git)
    cd AI-Engineer-Practice
    ```

2.  **(Opsional tapi direkomendasikan) Buat virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Di Windows gunakan: venv\Scripts\activate
    ```

3.  **Instal semua dependensi:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Jalankan Notebook:** Buka dan jalankan sel-sel di dalam file `quest-7.ipynb`.

---

## 🔮 Langkah Selanjutnya

Beberapa ide untuk pengembangan proyek ini di masa depan:
* Mencoba *fine-tuning* model Transformer yang lebih besar (seperti RoBERTa atau BERT-large).
* Melakukan analisis sentimen multi-kelas (misal: sangat positif, positif, netral, negatif, sangat negatif).
* Menerapkan model pada dataset ulasan dalam Bahasa Indonesia.

---

## 👤 Author

* **Nama:** Dava Ilham Muhammad
* **GitHub:** [@DavaIlham261](https://github.com/DavaIlham261)
* **LinkedIn:** [Dava Ilham Muhammad](https://www.linkedin.com/in/dava-muhammad-4861a3286/)