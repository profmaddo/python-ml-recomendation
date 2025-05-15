# 📚 Language Teacher Recommendation MVP

This project demonstrates a simple, local MVP for recommending language teachers using two popular Python libraries: **Surprise** (collaborative filtering) and **LightFM** (hybrid recommendation). It's built to run easily in environments like PyCharm or Jupyter Notebook with sample CSV files.

## 🚀 Features

- Load sample data for students, teachers, and ratings.
- Train and evaluate a recommendation model using:
  - 🎯 Surprise (SVD, collaborative filtering)
  - 🧠 LightFM (hybrid filtering with metadata)
- Print personalized top-5 teacher recommendations.
- Evaluate model accuracy using RMSE (Surprise) and precision@5 (LightFM).

---

## 📦 Requirements

Create a virtual environment and install the following packages:

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

pip install pandas numpy
pip install scikit-surprise
pip install lightfm
```

> ⚠️ If you use NumPy 2.x and see compatibility warnings with Surprise, run:
```bash
pip install numpy==1.24.4
```

---

## 📂 Files

| File | Description |
|------|-------------|
| `interacoes.csv` | Sample student-teacher ratings |
| `professores.csv` | List of teachers with metadata |
| `alunos.csv` | Optional list of student profiles |
| `svd_recomendador.py` | Uses Surprise with SVD algorithm |
| `lightfm_recomendador.py` | Uses LightFM with item features |

---

## 🧪 How to Run

### 1. Run Surprise SVD
```bash
python svd_recomendador.py
```
Expected output:
```
RMSE: 0.97
Top 5 recommended teachers for aluno1:
- prof2
- prof5
- ...
```

### 2. Run LightFM hybrid model
```bash
python lightfm_recomendador.py
```
Expected output:
```
Precision@5: 0.55
Top 5 recommended teachers for aluno1:
- prof4
- prof10
- ...
```

---

## 🧠 Why both?

| Feature | Surprise | LightFM |
|---------|----------|---------|
| Explicit feedback (ratings) | ✅ | ✅ |
| Implicit feedback (likes/clicks) | ❌ | ✅ |
| Use of metadata (e.g., language, level) | ❌ | ✅ |
| Simplicity | ✅ | ➖ |
| Scalability | ➖ | ✅ |

---

## 🤝 Contributing

Feel free to fork this repository, explore the logic, test with your own dataset, or connect it to an API or frontend. Your feedback and ideas are welcome!

---

## ✉️ Feedback

If you’ve had any experience with support operations for personalization and recommendation systems, what did you think of this project? Leave a comment or open an issue.

