# Offensive Language Detection System

This project is an end-to-end machine learning solution for detecting offensive or hate speech in text data. It leverages a deep learning model and natural language processing techniques to classify whether a given sentence contains offensive content.

##  Features

- Detects offensive or hate speech in real-time user input
- Trained on labeled dataset with a neural network
- Language detection and preprocessing with Keras tokenizer
- Easy-to-use Flask interface for inference

##  Technologies Used

- Python
- Keras / TensorFlow
- Flask
- Natural Language Toolkit (NLTK)
- Scikit-learn
- Pandas, NumPy

##  Dataset

The model is trained using a CSV dataset (`train.csv`) containing labeled text samples marked as offensive or non-offensive. You can customize or expand the dataset as needed for better accuracy.

## ⚙ How to Run the Project

1. **Clone the repository**
```bash
git clone https://github.com/prasanna909/offensive-detection.git
cd offensive-detection

2. Create and activate a virtual environment (recommended)

  python -m venv venv
  venv\Scripts\activate

3. Install dependencies

  pip install -r requirements.txt

4. Run the app
   python app.py

5. Open in browser
   http://127.0.0.1:5000/


