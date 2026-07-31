# Obesity Levels Prediction Dashboard

Interactive Streamlit coursework dashboard for the obesity-level dataset. It provides:

- sidebar filters and downloadable filtered data;
- obesity, numeric, relationship, and lifestyle charts;
- a fair hold-out comparison of KNN, logistic regression, decision tree, and random forest;
- an individual obesity-level prediction form with class probabilities.

## Run it

```powershell
pip install -r requirements.txt
streamlit run app.py
```

The CSV is included in `data/`. You can also upload another CSV using the same column structure in the sidebar.
