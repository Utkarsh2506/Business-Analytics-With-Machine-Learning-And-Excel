
# Business Analytics with Machine Learning and Excel

This project demonstrates how machine learning can be applied to business analytics using Excel datasets, Flask for the web interface, and pre-trained models to predict wine quality.

## 📂 Project Structure

```
Business-Analytics-With-ML-And-Excel/
├── app.py                          # Flask application
├── Model.py                        # ML model logic
├── Wine Company.xlsx               # Excel dataset
├── wine_quality_red_model.pkl      # Trained model (Red wine)
├── wine_quality_white_model.zip    # Trained model (White wine)
├── static/
│   └── style.css                   # Frontend styling
├── templates/
│   ├── index.html                  # Web UI
│   └── favicon.ico
├── DC PROJECT REPORT.pdf           # Documentation/report
```

## 🚀 How to Run the App

1. **Install dependencies**:
   ```bash
   pip install flask pandas scikit-learn openpyxl
   ```

2. **Run the Flask server**:
   ```bash
   python app.py
   ```

3. **Access the app**:
   Open your browser and go to `http://127.0.0.1:5000`

## 🔍 Features

- Upload wine sample data.
- Predict quality using trained ML models (Red and White wine).
- Visual representation via HTML frontend.
- Business analytics insights based on Excel data.

## 📊 Technologies Used

- Python, Flask
- Pandas, scikit-learn
- HTML/CSS (Jinja templates)
- Excel (Data source)

## 📄 Project Report

For detailed methodology and results, see [`DC PROJECT REPORT.pdf`](./DC%20PROJECT%20REPORT.pdf).

---

## ✨ Credits

Developed as part of a Business Analytics and Machine Learning project.
