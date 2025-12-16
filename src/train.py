import re
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
import pickle

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


DATA_URL = "https://breathecode.herokuapp.com/asset/internal-link?id=932&path=url_spam.csv"
MODEL_PATH = "models/svm_url_spam.pkl"

#Preprocesado de URLs
STOPWORDS = set(stopwords.words("english")) 
LEMMATIZER = WordNetLemmatizer()

def url_tokenizer(url: str):
    url = str(url).lower()
    raw_tokens = re.split(r"[^a-z0-9]+", url)

    tokens = []
    for t in raw_tokens:
        if len(t) < 2:
            continue
        if t in STOPWORDS:
            continue
        tokens.append(LEMMATIZER.lemmatize(t))

    return tokens


def main():
    # 1) Cargar dataset
    df = pd.read_csv(DATA_URL)

    # Inspección rápida
    print(df.head())
    print(df.columns)


    url_col = "url" if "url" in df.columns else df.columns[0]
    y_col = "is_spam" if "is_spam" in df.columns else df.columns[-1]

    X = df[url_col].astype(str)
    y = df[y_col].astype(int)

    # 2) Split train/test (estratificado)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2025, stratify=y
    )

    # 3) Pipeline base: TF-IDF + Linear SVM
    base_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            tokenizer=url_tokenizer,
            token_pattern=None,
            ngram_range=(1, 2),
            min_df=2
        )),
        ("svm", LinearSVC())
    ])

    print("== Entrenando SVM (default) ==")
    base_pipeline.fit(X_train, y_train)

    y_pred = base_pipeline.predict(X_test)
    print("\n== Resultados (modelo base) ==")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, digits=4))

    # 4) GridSearch sobre hiperparámetros útiles
param_grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df": [1, 2, 5],
        "svm__C": [0.1, 1, 3, 10],
        "svm__loss": ["hinge", "squared_hinge"],
        "svm__class_weight": [None, "balanced"],
    }

grid = GridSearchCV(
    base_pipeline,
    param_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1,
    verbose=1
    )

print("\n== Ejecutando GridSearch ==")
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("\n== Mejor configuración ==")
print(grid.best_params_)

y_pred_best = best_model.predict(X_test)
print("\n== Resultados mejor modelo ==")
print(confusion_matrix(y_test, y_pred_best))
print(classification_report(y_test, y_pred_best, digits=4))

    # 5) Guardar modelo
with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)

print(f"\nModelo guardado en: {MODEL_PATH}")


if __name__ == "__main__":
    main()


#Predicción
MODEL_PATH = "models/svm_url_spam.pkl"

def predict(url: str):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return int(model.predict([url])[0])


if __name__ == "__main__":
    urls = [
        "http://free-gift-cards-now.com/win",
        "https://www.wikipedia.org/wiki/Support_vector_machine",
    ]

    for u in urls:
        print(u, "=> spam" if predict(u) == 1 else "no spam")
