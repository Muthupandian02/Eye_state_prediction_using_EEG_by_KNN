# 👁️ Eye State Prediction using EEG Signals

> Predicting whether a person's eyes are open or closed — directly from brainwave data — using classical ML models, systematic evaluation, and a tuned KNN classifier deployed via FastAPI.


## 🧠 What This Project Is About

The human brain produces distinct electrical patterns depending on whether the eyes are open or closed. This project taps into those patterns using **EEG (Electroencephalography)** data — 14 electrode readings per timestep — and trains machine learning models to classify the **eye state** in real time.

This isn't just a classification exercise. The journey here involved comparing seven models, narrowing down to the top three, tuning the best one with hyperparameter optimization, and finally wrapping it in a clean **FastAPI web app** ready for deployment.


## 📦 Dataset

- **Source:** UCI Machine Learning Repository — [EEG Eye State Dataset](https://archive.ics.uci.edu/ml/datasets/EEG+Eye+State)
- **Samples:** 14,980 instances
- **Features:** 14 EEG electrode values (AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4)
- **Target:** Eye state — `0` (Eyes Open) or `1` (Eyes Closed)


## 🔬 Model Comparison — The First Pass

Before jumping to the best model, all major classifiers were benchmarked. Here's how they stacked up:

| Model                  | Train F1 | Test F1 | Notes                          |
|------------------------|----------|---------|--------------------------------|
| Random Forest          | 1.0000   | 0.8464  | Overfits, but decent test      |
| Gradient Boosting      | 0.7609   | 0.7242  | Weaker overall                 |
| Logistic Regression    | 0.6233   | 0.6127  | Underfits                      |
| Decision Tree          | 1.0000   | 0.7382  | Overfits, weaker test          |
| XGBoost                | 0.9734   | 0.8592  | Strong balance                 |
| SVC                    | 0.8619   | 0.8447  | Consistent                     |
| **KNeighborsClassifier** | **0.9410** | **0.8989** | **Best generalization** ✅ |

**Takeaway:** Models that memorized training data (Random Forest, Decision Tree with perfect Train F1) didn't hold up well on unseen data. KNN struck the best balance between learning and generalizing.


## 🏆 Top 3 Models — Class-wise Evaluation

After the first round, the top three candidates — **XGBoost**, **KNN**, and **SVC** — were evaluated more closely using **class-wise F1 scores** (positive = eyes closed, negative = eyes open):

| Model                | Positive F1 | Negative F1 | Balance   | Notes                   |
|----------------------|-------------|-------------|-----------|-------------------------|
| XGBoost              | 0.859       | 0.884       | Good      | Strong, balanced        |
| **KNeighborsClassifier** | **0.899** | **0.915** | **Excellent** | **Best performer** ✅ |
| SVC                  | 0.845       | 0.874       | Moderate  | Lower recall            |

KNN consistently outperformed across both classes, making it the clear choice for hyperparameter tuning.


## ⚙️ Hyperparameter Tuning — KNN

The KNN classifier was tuned using **GridSearchCV** across key parameters:

- `n_neighbors` — number of nearest neighbors
- `weights` — uniform vs. distance-based
- `metric` — euclidean, manhattan, minkowski

### Final Tuned Model Performance

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 0.9119 |
| Precision | 0.9027 |
| Recall    | 0.9047 |
| F1 Score  | 0.9037 |

With **91% accuracy** and consistent precision-recall balance, the tuned KNN model is both reliable and interpretable.


## 🌐 Deployment — FastAPI Web App

The trained KNN model is saved using `joblib` and served through a **FastAPI** backend with three HTML pages:

| Page                  | Purpose                                                  |
|-----------------------|----------------------------------------------------------|
| `home.html`           | Landing page with project overview and navigation        |
| `description.html`    | Explains the dataset, methodology, and model details     |
| `prediction.html`     | Input form to submit EEG values and get a live prediction |

### Running the App Locally

```bash
# 1. Clone the repository
git clone https://github.com/Muthupandian02/Eye_state_prediction_using_EEG_by_KNN.git
cd Eye_state_prediction_using_EEG_by_KNN

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the FastAPI server
uvicorn main:app --reload

# 4. Open in browser
http://127.0.0.1:2000
```

---

## 🗂️ Project Structure

```
Eye_state_prediction_using_EEG_by_KNN/
│
├── image/
│   └── prediction finalized.mp4
│   └── eeg_electrode.png
│   └── image-1.png
│   └── image.png
│
├── notebook/
│   └── task.ipynb          # Raw dataset
│   └── DS Mini Project Guidelines.docx
│
├── pickle_files/
│   └── knn_model.pkl              # Saved KNN model (joblib)
│   └── scaler.pkl
│
├── image/
│   └── boxplot1.png
│   └── boxplot.png
│   └── kdeplot.png
│   └── kdeplot2.png
│   └── pairplot.png
│   └── piechart.png
│
├── templates/
│   ├── home.html                  # Landing page
│   ├── description.html           # Project description
│   └── prediction.html            # Prediction interface
│
├── app.py                        # FastAPI application
├── model_training.ipynb           # EDA, training, evaluation notebook
├── requirements.txt               # Python dependencies
└── README.md
```


## 🛠️ Tech Stack

| Category        | Tools / Libraries                                        |
|-----------------|----------------------------------------------------------|
| Language        | Python 3.x                                               |
| ML Models       | Scikit-learn (KNN, SVC, RF, DT, LR, GB), XGBoost        |
| Tuning          | GridSearchCV                                             |
| Model Saving    | Joblib                                                   |
| Web Framework   | FastAPI + Uvicorn                                        |
| Frontend        | HTML, CSS (Jinja2 templates)                             |
| Data Handling   | Pandas, NumPy                                            |
| Visualization   | Matplotlib, Seaborn                                      |


## 💡 Key Takeaways

- **KNN works surprisingly well** for EEG-based classification when the feature space is well-scaled and the neighborhood size is tuned.
- **Overfitting is real** — models with Train F1 of 1.0 (Random Forest, Decision Tree) underperformed on test data by a noticeable margin.
- **Class balance matters** — class-wise F1 evaluation revealed SVC's weakness in recall, which aggregate accuracy would have hidden.
- **Deployment readiness** — saving the model and exposing it through a FastAPI endpoint makes this production-ready, not just a notebook experiment.

## 👨‍💻 Author

**Muthupandian**
Biomedical Engineering Graduate | AI/ML Enthusiast
📍 Tamil Nadu, India
🔗 [GitHub](https://github.com/Muthupandian02)


## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

*If this project helped you, consider giving it a ⭐ — it keeps the motivation alive!*
