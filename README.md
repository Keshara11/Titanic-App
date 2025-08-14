# Titanic Survival Prediction App

A Streamlit-based machine learning web application that predicts the survival chances of Titanic passengers based on user input or dataset features.  
This project uses the famous [Titanic dataset](https://www.kaggle.com/c/titanic) to train and deploy a prediction model, allowing interactive exploration, visualization, and predictions.

---

## 🚀 Features

- **Dataset Overview**:  
  View dataset shape, columns, data types, and sample rows.
  
- **Interactive Data Filtering**:  
  Filter passengers by gender, class, age range, embarkation port, etc.

- **Data Visualization**:  
  Generate charts such as:
  - Survival rate by gender
  - Survival rate by passenger class
  - Age distribution of survivors vs non-survivors

- **Machine Learning Model**:  
  - Trained on Titanic dataset  
  - Generates predictions for passenger survival based on:
    - `Pclass`
    - `Sex`
    - `Age`
    - `SibSp`
    - `Parch`
    - `Fare`
    - `Embarked`
  - Model saved as `model.pkl` and loaded in the app

- **Live Deployment**:  
  Access the app here: [Titanic App on Streamlit Cloud](https://titanic-app-qmewfslcchpfdwxmay9jql.streamlit.app/)

---

## 📂 Project Structure

```

.
├── app.py                # Main Streamlit app script
├── model.pkl             # Trained machine learning model
├── requirements.txt      # Python dependencies
├── data/                 # Dataset files (if included locally)
├── notebooks/            # Jupyter notebooks for data exploration & model training
└── README.md             # Project documentation

````

---

## 🛠 Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Keshara11/Titanic-App.git
   cd Titanic-App
````

2. **Create and activate a virtual environment** (optional but recommended)

   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app locally**

   ```bash
   streamlit run app.py
   ```

---

## 📊 Model Training

The model was trained using:

* **Scikit-learn**: Logistic Regression / RandomForestClassifier
* **Data preprocessing**: Handling missing values, encoding categorical variables, feature scaling
* **Evaluation**: Accuracy, precision, recall, F1-score

Training steps can be found in the `notebooks/` folder.

---

## 📸 Screenshots

**Home Page**
![Home](screenshots/home.png)

**Prediction Form**
![Prediction](screenshots/predict.png)

**Visualizations**
![Visualizations](screenshots/visuals.png)

---

## 🤝 Contributing

1. Fork the repository
2. Create a new branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m 'Add new feature'`
4. Push to your branch: `git push origin feature-name`
5. Submit a Pull Request

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Rukmi Keshara**
GitHub: [@Keshara11](https://github.com/Keshara11)

```

