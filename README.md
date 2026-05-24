# 📧 NLP Email Summarizer

An end-to-end NLP pipeline that summarizes email threads using both **extractive** (TF-IDF / TextRank) and **abstractive** (BART) techniques, with full evaluation via ROUGE scores and an interactive Gradio UI.

---

## Features

- **Extractive Summarization** — TextRank-style ranking using TF-IDF cosine similarity to select the most informative sentences from an email.
- **Abstractive Summarization** — Generates fluent, human-like summaries using `facebook/bart-large-cnn` from Hugging Face Transformers.
- **ROUGE Evaluation** — Benchmarks both methods against reference summaries using ROUGE-1, ROUGE-2, and ROUGE-L metrics.
- **Exploratory Data Analysis (EDA)** — Visualizes length distributions, compression ratios, word clouds, and top word frequencies.
- **Interactive UI** — A Gradio web app for real-time email summarization with configurable parameters.

---

## Dataset

The notebook uses the [Email Thread Summary Dataset](https://www.kaggle.com/datasets/marawanxmamdouh/email-thread-summary-dataset) from Kaggle, which contains two CSV files:

- `email_thread_details.csv` — email thread bodies (`body` column)
- `email_thread_summaries.csv` — reference summaries (`summary` column)

Both files are merged on `thread_id` to form the working dataframe.

---

## Setup

### 1. Install Dependencies

```bash
pip install kaggle transformers sumy rouge-score nltk gradio matplotlib seaborn wordcloud pandas numpy scikit-learn torch sentencepiece
```

### 2. Configure Kaggle API

Upload your `kaggle.json` API key when prompted (the notebook uses `google.colab.files.upload()`), or place it manually at `~/.kaggle/kaggle.json` with permissions `600`.

### 3. Download the Dataset

```bash
kaggle datasets download -d marawanxmamdouh/email-thread-summary-dataset --unzip -p ./data
```

---

## Project Structure

```
.
├── email_summarizer_nlp.ipynb   # Main notebook
├── data/
│   ├── email_thread_details.csv
│   └── email_thread_summaries.csv
└── outputs/
    ├── eda_length_distribution.png
    ├── eda_statistical_overview.png
    ├── eda_wordclouds.png
    ├── eda_top_words.png
    ├── eda_stats_table.png
    ├── rouge_evaluation.png
    ├── evaluation_results.csv
    └── manual_summary.txt       # Saved from the Gradio UI
```

---

## Notebook Walkthrough

### 1. Data Loading & Cleaning
- Loads and merges both CSVs on `thread_id`
- Drops nulls, strips whitespace, filters out emails shorter than 50 characters

### 2. Exploratory Data Analysis
Generates 5 figures saved to `./outputs/`:

| Figure | Description |
|--------|-------------|
| `eda_length_distribution.png` | Histograms of email/summary word counts and compression ratio |
| `eda_statistical_overview.png` | Box plots and email-length vs summary-length scatter plot |
| `eda_wordclouds.png` | Word clouds for email bodies and summaries |
| `eda_top_words.png` | Top 20 most frequent words in each |
| `eda_stats_table.png` | Descriptive statistics table |

### 3. Extractive Summarization (TF-IDF / TextRank)

```python
extractive_summarize(text, num_sentences=3)
```

Tokenizes the email into sentences, builds a TF-IDF matrix, computes cosine similarity between sentences, and ranks them using a PageRank-style score. Returns the top `num_sentences` in original order.

### 4. Abstractive Summarization (BART)

```python
abstractive_summarize(text, max_len=130, min_len=30)
```

Uses `facebook/bart-large-cnn` to generate a fluent abstractive summary. Automatically truncates input to 900 words and runs on GPU if available.

### 5. ROUGE Evaluation

Evaluates both methods on up to 50 sampled emails against reference summaries. Results are saved to `evaluation_results.csv` and a comparison chart is exported.

### 6. Gradio UI

Launches an interactive web app with:
- A text box to paste an email
- 3 preloaded sample emails
- Method selection: Extractive, Abstractive, or Both
- Sliders for number of sentences, max/min output length
- Live stats (word count, compression ratio)
- Save button to export results

---

## Usage (Gradio App)

After running all cells, the final cell launches the app:

```
Running on public URL: https://xxxx.gradio.live
```

Open the URL, paste an email, select your method, and click **Summarize**.

---

## Model

| Component | Details |
|-----------|---------|
| Abstractive model | `facebook/bart-large-cnn` |
| Tokenizer | `AutoTokenizer` from Hugging Face |
| Device | CUDA (GPU) if available, else CPU |
| Beam search | 4 beams, early stopping |

---

## Requirements

- Python 3.8+
- Google Colab (recommended for GPU) or local environment with CUDA
- Kaggle API key

---

## Outputs

| File | Description |
|------|-------------|
| `evaluation_results.csv` | Per-sample ROUGE scores for both methods |
| `rouge_evaluation.png` | Bar chart and histogram of ROUGE scores |
| EDA plots (5 `.png` files) | Visualizations of the dataset |
| `manual_summary.txt` | Summary saved from the Gradio UI |
