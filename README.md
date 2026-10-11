# 📧 Phishing Email Detection Using Machine Learning & NLP

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()

An intelligent cybersecurity defense system applying Natural Language Processing (NLP) and Machine Learning to classify incoming emails as **Safe** or **Phishing** with **97.75% accuracy**. Trained on real-world email datasets containing thousands of legitimate and malicious communications.

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Model Performance & Evaluation](#-model-performance--evaluation)
- [System Architecture & Workflow](#-system-architecture--workflow)
- [Repository Structure](#-repository-structure)
- [Installation & Setup](#-installation--setup)
- [Quickstart: Testing Emails](#-quickstart-testing-emails)
- [Dataset Information](#-dataset-information)
- [Project Documentation](#-project-documentation)
- [Author & Acknowledgements](#-author--acknowledgements)

---

## 🔍 Project Overview

Phishing remains the #1 initial attack vector in cyber attacks, credential harvesting, and financial fraud. Attackers use social engineering, deceptive links, and urgency cues to deceive recipients.

This project delivers an automated email inspection pipeline that:
- Normalizes raw email text (HTML tag stripping, tokenizing URLs into generic `url` anchors and email addresses into `email` tokens).
- Converts text into numerical vectors using **TF-IDF Vectorization** (capturing high-frequency spam and phishing triggers).
- Evaluates **Logistic Regression**, **Multinomial Naive Bayes**, and **Random Forest** classifiers.
- Provides immediate real-time prediction with probability confidence scores.

---

## 📊 Model Performance & Evaluation

The models were evaluated using Accuracy, Precision, Recall, and F1-Score:

| Model | Accuracy | Precision | Recall | F1 Score | Status |
|---|:---:|:---:|:---:|:---:|---|
| **Logistic Regression (Selected)** | **97.75%** | **97.98%** | **95.96%** | **96.96%** | 🏆 **Best Overall Balance** |
| **Multinomial Naive Bayes** | 97.41% | 97.44% | 95.58% | 96.50% | Fast probabilistic baseline |
| **Random Forest** | 97.23% | 95.44% | 97.26% | 96.34% | High recall ensemble trees |

> **Selection**: Logistic Regression attained the highest accuracy (**97.75%**) and precision (**97.98%**), ensuring minimal false positives while catching malicious emails.

---

## ⚙️ System Architecture & Workflow

```
Incoming Email (Subject & Body)
              │
              ▼
    [ Text Preprocessing ]
      ├── Convert to lowercase
      ├── Strip HTML tags (<...>)
      ├── Replace URLs -> 'url'
      ├── Replace emails -> 'email'
      └── Punctuation removal
              │
              ▼
    [ TF-IDF Vectorizer ]
      └── Transform into sparse numerical feature vector
              │
              ▼
    [ Logistic Regression Classifier ]
      └── Evaluates fraud & urgency markers
              │
              ▼
    [ Classification Output ]
      ├── Safe Email (Legitimate communication)
      └── Phishing Email (Malicious alert with confidence score)
```

---

## 📁 Repository Structure

```
Phishing-Email-Detection/
├── docs/
│   └── Phishing_Email_Detection_Report.pdf   # Complete project documentation & analysis
├── models/
│   ├── phishing_model.pkl                   # Trained Logistic Regression classifier
│   └── tfidf_vectorizer.pkl                 # TF-IDF feature extractor
├── notebooks/
│   └── phishing_email_detection.ipynb       # Jupyter Notebook with EDA, training & metrics
├── .gitignore                               # Ignores large datasets & cache files
├── predict.py                               # Interactive prediction utility
├── README.md                                # Project documentation
└── requirements.txt                         # Python dependencies
```

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/<YOUR_USERNAME>/Phishing-Email-Detection.git
cd Phishing-Email-Detection
```

### 2. Create a Virtual Environment (Optional)
```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# macOS / Linux:
source venv/bin/activate
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

---

## ⚡ Quickstart: Testing Emails

Test any email message without re-training:

### Interactive Mode:
```bash
python predict.py
```
*Enter any email subject or body when prompted to receive instantaneous classification.*

### Single-Command Mode:
```bash
python predict.py "URGENT: Your account has been temporarily locked. Click http://bank-secure-login.xyz to verify your identity."
```

**Sample Output:**
```
============================================================
      📧 PHISHING EMAIL DETECTION SYSTEM (ML Inference)      
============================================================
✓ Model and TF-IDF Vectorizer loaded successfully.

Input: URGENT: Your account has been temporarily locked...
Prediction: [PHISHING]
Confidence -> Safe: 1.15% | Phishing: 98.85%
```

---

## 📚 Dataset Information

- **Dataset**: Phishing Email Dataset (`Phishing_Email.csv`)
- Contains thousands of labeled emails categorized as legitimate communications or phishing attempts.
- *Raw dataset CSV is excluded from Git tracking due to size limits; place `Phishing_Email.csv` in the root folder if executing the full Jupyter notebook.*

---

## 📄 Project Documentation

For comprehensive research methodology, feature weights, and algorithmic justification, see:
- [`docs/Phishing_Email_Detection_Report.pdf`](docs/Phishing_Email_Detection_Report.pdf)

---

## 👨‍💻 Author & Acknowledgements

- **Department / Course**: Information & Communication Technology (IICT)
- Developed using Python, Scikit-Learn, and NLP methodologies.
