
# Business Analytics with Machine Learning and Excel

This project demonstrates how machine learning can be applied to business analytics using Excel datasets, Flask for the web interface, and pre-trained models to predict wine quality.

## 📂 Project Structure

```
Business-Analytics-With-ML-And-Excel/
├── app.py                             # Flask application
├── Model.py                           # ML model logic
├── Wine Company.xlsx                  # Excel dataset
├── wine_quality_red_model.pkl         # Trained model (Red wine)
├── wine_quality_white_model.zip       # Trained model (White wine)
├── static/
│   └── style.css                      # Frontend styling
├── templates/
│   ├── index.html                     # Web UI
│   └── favicon.ico
├── Wine Company Project Report.pdf    # Documentation/report
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

- Wine Quality Prediction:
     Predict red and white wine quality based on physicochemical properties using trained ML models.
- Excel Business Models:
   - Transportation cost optimization using Excel Solver.
   - Profitability analysis for different wine types.
   - NPV (Net Present Value) calculations for long-term investment decisions.
- Interactive Web Interface:
     Clean, responsive UI with form validation, autofill for best quality wine, and real-time predictions.

## 📊 Technologies Used

- Programming & Frameworks: Python, Flask
- Business Analysis: Excel (Solver, NPV, Linear Programming, Optimization Models)
- Data Visualisation: Pandas, scikit-learn
- Frontend: HTML, CSS (Bootstrap), Jinja2 Templates



## 📄 Project Report

For detailed methodology and results, see [`Wine Company Project Report`]().

---

## ✨ Credits

Developed as part of a Business Analytics and Machine Learning project.
