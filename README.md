# Synthetic Financial Transaction Data Generator using Conditional GAN

## 🚀 Live Demo

**[Try the interactive app on Streamlit →](https://cgansynthetic-data-generator-6byxvzuiph4ewfvodtc92i.streamlit.app/)**

Generate realistic synthetic fraud and non-fraud financial transactions instantly. No setup needed—just click and explore.

---

## 📌 What This Does

This project **generates realistic synthetic financial transaction data** using a Conditional GAN (CGAN). 

**Why this matters:**
- Banks and fintech companies need fraud detection training data
- Real financial data is private and hard to access
- Synthetic data solves this: realistic enough to train ML models, but completely fake
- No privacy issues, no compliance headaches

**In 30 seconds:** Upload a small dataset → Model learns fraud patterns → Generates thousands of realistic fake transactions → Use for training fraud detectors

---

## ✨ Key Features

- ✅ **Fully Trained CGAN Model** - Pre-trained on IEEE-CIS Fraud Detection dataset
- ✅ **Interactive Streamlit Interface** - Generate synthetic transactions with one click
- ✅ **Real Fraud Patterns** - Learns and replicates actual fraud behavior
- ✅ **Privacy-Safe** - Zero real customer data exposed
- ✅ **Production Ready** - Model weights included, deployable anywhere

---

## 📊 How It Works

1. **Data Preprocessing** - Real fraud transactions are cleaned, normalized, and prepared
2. **CGAN Training** - Generator learns to create fake transactions that mimic real fraud patterns
3. **Synthetic Data Generation** - Model outputs realistic transaction data with proper fraud distribution
4. **Quality Validation** - Generated data is evaluated against real data characteristics

---

## 🛠️ Tech Stack

**Core:**
- Python 3.9+
- PyTorch - Deep learning framework
- Pandas & NumPy - Data manipulation

**Deployment:**
- Streamlit - Interactive web interface
- Streamlit Cloud - Hosting

**ML/Data:**
- Scikit-learn - Feature scaling
- Pandas - Data processing

---

## 📁 Project Structure

```
cgan_synthetic_data_generator/
├── app.py                      # Streamlit app (main interface)
├── gan_model.py                # CGAN architecture (Generator + Discriminator)
├── train.py                    # Training pipeline
├── data_preprocessing.py        # Data cleaning and feature scaling
├── fraud_classifier.py          # Evaluate synthetic data quality
├── evaluate.py                 # Model evaluation metrics
├── generator.pth               # Trained generator weights
├── discriminator.pth           # Trained discriminator weights
├── requirements.txt            # Python dependencies
└── README.md
```

---

## 🚀 Quick Start

### Option 1: Use the Live Demo (Easiest)
Just visit the **[Streamlit link above](https://cgansynthetic-data-generator-6byxvzuiph4ewfvodtc92i.streamlit.app/)** and start generating data.

### Option 2: Run Locally

```bash
# Clone repo
git clone https://github.com/FAKE-SURYA/cgan_synthetic-data-generator.git
cd cgan_synthetic-data-generator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 📈 Results

The trained model successfully:
- ✅ Generates 1000+ synthetic transactions in seconds
- ✅ Maintains realistic fraud-to-legitimate ratio (~3% fraud)
- ✅ Captures feature distributions of real data
- ✅ Produces valid transactions ready for ML training

**Example Output:**
- Input: 100 real transactions
- Output: 10,000 synthetic transactions with identical statistical properties

---

## 🎓 What I Learned

- Building and training GANs from scratch (no pre-trained models)
- Handling imbalanced financial data
- Conditional generation for multiple classes
- Deploying ML models as interactive web apps
- Privacy-preserving synthetic data generation

---

## 📚 Resources Used

- IEEE-CIS Fraud Detection Dataset (Kaggle)
- PyTorch Documentation
- Streamlit Framework
- GAN Papers and tutorials

---

## 🔄 What's Next

Potential improvements:
- Fine-tune model for other transaction types (e-commerce, crypto, etc.)
- Add feature importance analysis
- Build REST API for batch generation
- Compare synthetic vs real data in fraud detection models

---

## 👨‍💻 Author

**Surya Pratap Singh**
- GitHub: [@FAKE-SURYA](https://github.com/FAKE-SURYA)
- LinkedIn: [surya-pratap-singh-2a](https://www.linkedin.com/in/surya-pratap-singh-2a)

---

## 📝 License

MIT License - Feel free to use and modify this project.
---

---

⭐ **If you found this useful, star the repo!**


