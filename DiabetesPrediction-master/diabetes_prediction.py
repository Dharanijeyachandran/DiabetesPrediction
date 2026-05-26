"""
Diabetes Prediction App (GUI) - Full script

Features:
- Trains ML models on pima_indians_diabetes.csv with preprocessing
- Uses SMOTE if available (to balance classes)
- GridSearchCV hyperparameter tuning (small grid for speed)
- Picks best model by test accuracy
- Front page (Name, Age, DOB) -> Next -> Prediction Window
- Each patient's results saved to Patient_Records/<Name_With_Underscores>.txt
- View patient history and pie chart
- Show parameter bar chart for current inputs

Author: ChatGPT (adapted for your project)
"""

import os
import pickle
import traceback
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tkinter import messagebox, ttk

# -------------------------------
# CONFIG
# -------------------------------
DATAFILE = "pima_indians_diabetes.csv"
PATIENT_DIR = "Patient_Records"
MODEL_FILE = "best_model.pkl"
SCALER_FILE = "scaler.pkl"
META_FILE = "model_meta.pkl"

os.makedirs(PATIENT_DIR, exist_ok=True)

# -------------------------------
# Helper: attempt optional imports
# -------------------------------
have_smote = False
have_xgboost = False
try:
    from imblearn.over_sampling import SMOTE

    have_smote = True
except Exception:
    print("imblearn not available — will skip SMOTE balancing (recommended: pip install imbalanced-learn)")

try:
    from xgboost import XGBClassifier

    have_xgboost = True
except Exception:
    print("xgboost not available — will skip XGBoost model (optional: pip install xgboost)")

# -------------------------------
# 1. Load and Preprocess Dataset
# -------------------------------
def load_and_preprocess(path=DATAFILE):
    df = pd.read_csv(path)
    # Copy to avoid chained-assignment issues
    df = df.copy()

    # Replace zeros in columns where zero is not a valid value
    cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_with_zero:
        # Treat zeros as missing
        df[col] = df[col].replace(0, np.nan)

    # Fill missing numeric columns with median (safe pandas usage)
    df = df.fillna(df.median(numeric_only=True))

    return df


# -------------------------------
# 2. Train models and pick best
# -------------------------------
def train_and_select_best(df):
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    # Optionally apply SMOTE for balancing
    if have_smote:
        try:
            sm = SMOTE(random_state=42)
            X_res, y_res = sm.fit_resample(X, y)
            print("SMOTE applied — classes balanced.")
        except Exception:
            print("SMOTE failed — falling back to original data.")
            X_res, y_res = X, y
    else:
        X_res, y_res = X, y

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)

    # Scaling
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Models + small hyperparameter grids
    models = {}
    results = {}

    # Logistic Regression
    lr = LogisticRegression(max_iter=500)
    lr_params = {'C': [0.1, 1, 10], 'solver': ['liblinear', 'lbfgs']}
    lr_grid = GridSearchCV(lr, lr_params, cv=3, n_jobs=-1)
    lr_grid.fit(X_train_s, y_train)
    models['LogisticRegression'] = lr_grid.best_estimator_
    results['LogisticRegression'] = accuracy_score(y_test, lr_grid.best_estimator_.predict(X_test_s))

    # SVM
    svm = SVC(probability=True)
    svm_params = {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear'], 'gamma': ['scale']}
    svm_grid = GridSearchCV(svm, svm_params, cv=3, n_jobs=-1)
    svm_grid.fit(X_train_s, y_train)
    models['SVM'] = svm_grid.best_estimator_
    results['SVM'] = accuracy_score(y_test, svm_grid.best_estimator_.predict(X_test_s))

    # Random Forest
    rf = RandomForestClassifier(random_state=42)
    rf_params = {'n_estimators': [100, 200], 'max_depth': [None, 6, 8]}
    rf_grid = GridSearchCV(rf, rf_params, cv=3, n_jobs=-1)
    rf_grid.fit(X_train_s, y_train)
    models['RandomForest'] = rf_grid.best_estimator_
    results['RandomForest'] = accuracy_score(y_test, rf_grid.best_estimator_.predict(X_test_s))

    # XGBoost (optional)
    if have_xgboost:
        try:
            xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
            xgb_params = {'n_estimators': [100, 200], 'max_depth': [3, 5]}
            xgb_grid = GridSearchCV(xgb, xgb_params, cv=3, n_jobs=-1)
            xgb_grid.fit(X_train_s, y_train)
            models['XGBoost'] = xgb_grid.best_estimator_
            results['XGBoost'] = accuracy_score(y_test, xgb_grid.best_estimator_.predict(X_test_s))
        except Exception as e:
            print("XGBoost training failed — skipping. Error:", e)

    # Select best by accuracy
    best_name = max(results, key=results.get)
    best_model = models[best_name]
    best_acc = results[best_name]
    print("Model accuracies:", results)
    print(f"Best model: {best_name} (accuracy={best_acc:.4f})")

    # Save model & scaler & meta
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(best_model, f)
    with open(SCALER_FILE, "wb") as f:
        pickle.dump(scaler, f)
    meta = {'model_name': best_name, 'accuracies': results, 'feature_names': list(X.columns)}
    with open(META_FILE, "wb") as f:
        pickle.dump(meta, f)

    return best_model, scaler, meta, (X_test_s, y_test, results)


# -------------------------------
# 3. Train at startup (fast prints)
# -------------------------------
try:
    print("Loading and preprocessing dataset...")
    df = load_and_preprocess(DATAFILE)
    print("Training models (this may take ~30-60s depending on your machine)...")
    best_model, scaler, meta, test_info = train_and_select_best(df)
except FileNotFoundError:
    traceback.print_exc()
    print(f"Dataset file '{DATAFILE}' not found. Please put '{DATAFILE}' in the same folder.")
    messagebox.showerror("Dataset Not Found", f"Dataset file '{DATAFILE}' not found. Place it next to this script.")
    raise SystemExit(1)
except Exception as e:
    traceback.print_exc()
    messagebox.showwarning("Training Error", f"Training encountered an error. Proceeding without trained model.\n\n{e}")
    # If training fails, try to fallback to a basic LogisticRegression trained on raw data quickly
    try:
        quick_df = load_and_preprocess(DATAFILE)
        Xq = quick_df.drop("Outcome", axis=1)
        yq = quick_df["Outcome"]
        scaler = StandardScaler()
        Xq_s = scaler.fit_transform(Xq)
        quick_lr = LogisticRegression(max_iter=500).fit(Xq_s, yq)
        best_model = quick_lr
        meta = {'model_name': 'QuickLogistic', 'accuracies': {}, 'feature_names': list(Xq.columns)}
        with open(MODEL_FILE, "wb") as f:
            pickle.dump(best_model, f)
        with open(SCALER_FILE, "wb") as f:
            pickle.dump(scaler, f)
        with open(META_FILE, "wb") as f:
            pickle.dump(meta, f)
        print("Fallback quick model trained.")
    except Exception:
        traceback.print_exc()
        print("Fatal: could not train fallback model. Exiting.")
        raise SystemExit(1)


# -------------------------------
# 4. GUI - Front page + Prediction window
# -------------------------------
def sanitize_filename(name: str) -> str:
    filename = name.strip().replace(" ", "_")
    # remove characters invalid for filenames
    return "".join(c for c in filename if c.isalnum() or c in ("_", "-")) + ".txt"


def open_prediction_window(patient_name, patient_age, patient_dob):
    # Main prediction window
    win = tk.Tk()
    win.title("🩺 Diabetes Prediction System")
    win.geometry("840x780")
    win.config(bg="#f0fbff")

    # vars
    name_var = tk.StringVar(value=patient_name)
    age_var = tk.StringVar(value=patient_age)
    dob_var = tk.StringVar(value=patient_dob)
    gender_var = tk.StringVar(value="Male")
    preg_var = tk.StringVar(value="0")
    glucose_var = tk.StringVar()
    bp_var = tk.StringVar()
    skin_var = tk.StringVar()
    insulin_var = tk.StringVar()
    bmi_var = tk.StringVar()
    dpf_var = tk.StringVar()
    age_input_var = tk.StringVar(value=patient_age)  # duplicate display

    # Hide pregnancy if Male
    def on_gender_change(event=None):
        if gender_var.get() == "Male":
            preg_label.grid_remove()
            preg_entry.grid_remove()
        else:
            preg_label.grid(row=2, column=0, padx=10, pady=5, sticky="e")
            preg_entry.grid(row=2, column=1, padx=10, pady=5)

    def predict():
        try:
            # Build input vector in the original dataset order:
            # ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
            inp = [
                float(preg_var.get() or 0),
                float(glucose_var.get()),
                float(bp_var.get()),
                float(skin_var.get() or 0),
                float(insulin_var.get() or 0),
                float(bmi_var.get()),
                float(dpf_var.get() or 0),
                float(age_input_var.get() or 0)
            ]
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numeric values for all parameters.")
            return

        # scale
        arr = np.array(inp).reshape(1, -1)
        try:
            arr_s = scaler.transform(arr)
        except Exception as e:
            messagebox.showwarning("Scaling Error", f"Scaling failed. Check scaler.\n\n{e}")
            arr_s = arr

        # predict
        pred = best_model.predict(arr_s)[0]
        proba = None
        try:
            if hasattr(best_model, "predict_proba"):
                proba = best_model.predict_proba(arr_s)[0][1]  # probability of positive
            elif hasattr(best_model, "decision_function"):
                proba = best_model.decision_function(arr_s)[0]
        except Exception:
            proba = None

        result_box.config(state="normal")
        result_box.delete("1.0", tk.END)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result_text = f"Patient: {name_var.get()}  |  Age: {age_input_var.get()}  |  Gender: {gender_var.get()}\n"
        result_text += f"Date & Time: {now}\n\n"

        if pred == 1:
            result_text += "Prediction: DIABETIC\n\n"
            result_text += "Suggested actions:\n"
            result_text += "- Consult a doctor for confirmation and treatment.\n- Follow low-sugar diet, increase exercise.\n- Monitor glucose regularly.\n"
        else:
            result_text += "Prediction: NON-DIABETIC\n\n"
            result_text += "Suggested actions:\n- Maintain healthy diet and exercise.\n- Regular check-ups to monitor.\n"

        if proba is not None:
            try:
                prob_display = float(proba)
                # if probability in [0,1], display as percent
                if 0 <= prob_display <= 1:
                    result_text += f"\nRisk Score (probability): {prob_display*100:.1f}%\n"
                else:
                    result_text += f"\nRisk Score (score): {prob_display:.3f}\n"
            except Exception:
                result_text += f"\nRisk Score: {proba}\n"

        result_box.insert(tk.END, result_text)
        result_box.config(state="disabled")

        # Save per-patient record
        fname = os.path.join(PATIENT_DIR, sanitize_filename(name_var.get()))
        with open(fname, "a", encoding="utf-8") as f:
            f.write(f"{now} | Gender={gender_var.get()} | Pregnancies={preg_var.get()} | Glucose={glucose_var.get()} | BP={bp_var.get()} | Skin={skin_var.get()} | Insulin={insulin_var.get()} | BMI={bmi_var.get()} | DPF={dpf_var.get()} | Age={age_input_var.get()} | Result={'Diabetic' if pred==1 else 'Non-Diabetic'}\n")

        messagebox.showinfo("Saved", f"Result saved to {fname}")

    def view_history_for_patient():
        fname = os.path.join(PATIENT_DIR, sanitize_filename(name_var.get()))
        hist_win = tk.Toplevel(win)
        hist_win.title("Patient History")
        hist_win.geometry("700x600")
        txt_frame = tk.Frame(hist_win)
        txt_frame.pack(fill="both", expand=True)
        scroll = tk.Scrollbar(txt_frame)
        scroll.pack(side="right", fill="y")
        txt = tk.Text(txt_frame, wrap="word", yscrollcommand=scroll.set, font=("Arial", 10))
        txt.pack(fill="both", expand=True)
        scroll.config(command=txt.yview)

        diabetic = non_diabetic = 0
        if os.path.exists(fname):
            with open(fname, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines:
                    txt.insert(tk.END, line)
                    if "Diabetic" in line and "Non-Diabetic" not in line:
                        diabetic += 1
                    elif "Non-Diabetic" in line:
                        non_diabetic += 1
        else:
            txt.insert("1.0", "No records for this patient yet.")

        txt.config(state="disabled")

        # Pie chart
        pie_frame = tk.Frame(hist_win)
        pie_frame.pack(pady=10, fill="x")
        total = diabetic + non_diabetic
        if total > 0:
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.pie([diabetic, non_diabetic], labels=["Diabetic", "Non-Diabetic"], autopct='%1.1f%%', startangle=90)
            ax.set_title("Patient History Distribution")
            canvas = FigureCanvasTkAgg(fig, master=pie_frame)
            canvas.draw()
            canvas.get_tk_widget().pack()
        else:
            tk.Label(pie_frame, text="No data to plot for this patient.", font=("Arial", 11)).pack()

    def show_parameters_graph():
        # plot bar chart of current parameter values
        try:
            labels = ['Glucose', 'BP', 'Skin', 'Insulin', 'BMI', 'DPF', 'Age']
            vals = [
                float(glucose_var.get()), float(bp_var.get()), float(skin_var.get() or 0),
                float(insulin_var.get() or 0), float(bmi_var.get()), float(dpf_var.get() or 0), float(age_input_var.get() or 0)
            ]
        except ValueError:
            messagebox.showerror("Error", "Enter numeric values to plot parameters.")
            return
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(labels, vals)
        ax.set_title(f"Health Parameters - {name_var.get()}")
        ax.set_ylabel("Value")
        plt.tight_layout()
        plt.show()

    # UI Layout
    title = tk.Label(win, text="🩺 Diabetes Prediction System", font=("Arial", 20, "bold"), bg="#006e8a", fg="white")
    title.pack(fill="x", pady=8)

    frm = tk.Frame(win, bg="#f0fbff")
    frm.pack(pady=10)

    # Basic info
    tk.Label(frm, text="Patient Name:", bg="#f0fbff").grid(row=0, column=0, padx=8, pady=5, sticky="e")
    tk.Entry(frm, textvariable=name_var, width=28).grid(row=0, column=1, padx=8, pady=5)

    tk.Label(frm, text="Gender:", bg="#f0fbff").grid(row=1, column=0, padx=8, pady=5, sticky="e")
    gender_cb = ttk.Combobox(frm, textvariable=gender_var, values=["Male", "Female"], state="readonly", width=25)
    gender_cb.grid(row=1, column=1, padx=8, pady=5)
    gender_cb.bind("<<ComboboxSelected>>", on_gender_change)

    preg_label = tk.Label(frm, text="Pregnancies:", bg="#f0fbff")
    preg_label.grid(row=2, column=0, padx=8, pady=5, sticky="e")
    preg_entry = tk.Entry(frm, textvariable=preg_var)
    preg_entry.grid(row=2, column=1, padx=8, pady=5)

    # parameters labels
    labels = ["Glucose:", "Blood Pressure:", "Skin Thickness:", "Insulin:", "BMI:", "Diabetes Pedigree Function:", "Age:"]
    vars_ = [glucose_var, bp_var, skin_var, insulin_var, bmi_var, dpf_var, age_input_var]

    for i, (lbl, varr) in enumerate(zip(labels, vars_), start=3):
        tk.Label(frm, text=lbl, bg="#f0fbff").grid(row=i, column=0, padx=8, pady=5, sticky="e")
        tk.Entry(frm, textvariable=varr).grid(row=i, column=1, padx=8, pady=5)

    # Buttons
    btn_frame = tk.Frame(win, bg="#f0fbff")
    btn_frame.pack(pady=12)

    tk.Button(btn_frame, text="Predict Diabetes", command=predict, bg="#1b9e77", fg="white", width=18).grid(row=0, column=0, padx=8)
    tk.Button(btn_frame, text="View Patient History", command=view_history_for_patient, bg="#f39c12", fg="white", width=18).grid(row=0, column=1, padx=8)
    tk.Button(btn_frame, text="Show Parameters Graph", command=show_parameters_graph, bg="#e74c3c", fg="white", width=18).grid(row=0, column=2, padx=8)
    tk.Button(btn_frame, text="Clear Fields", command=lambda: clear_all(), bg="#2c3e50", fg="white", width=12).grid(row=0, column=3, padx=8)

    # result box
    result_box = tk.Text(win, height=12, width=95, state="disabled", bg="white", fg="black", relief="sunken")
    result_box.pack(pady=8)

    def clear_all():
        for v in [preg_var, glucose_var, bp_var, skin_var, insulin_var, bmi_var, dpf_var]:
            v.set("")
        result_box.config(state="normal")
        result_box.delete("1.0", tk.END)
        result_box.config(state="disabled")

    win.mainloop()


# -------------------------------
# 5. FRONT PAGE GUI
# -------------------------------
def open_front_page():
    front = tk.Tk()
    front.title("Diabetes Prediction - Welcome")
    front.geometry("520x360")
    front.config(bg="#e8f6ff")

    tk.Label(front, text="🩺 Diabetes Prediction System", font=("Arial", 20, "bold"), bg="#0077a3", fg="white").pack(fill="x", pady=10)

    frm = tk.Frame(front, bg="#e8f6ff")
    frm.pack(pady=25)

    name_var = tk.StringVar()
    age_var = tk.StringVar()
    dob_var = tk.StringVar()

    tk.Label(frm, text="Patient Name:", bg="#e8f6ff").grid(row=0, column=0, pady=8, sticky="e")
    tk.Entry(frm, textvariable=name_var, width=28).grid(row=0, column=1)

    tk.Label(frm, text="Age:", bg="#e8f6ff").grid(row=1, column=0, pady=8, sticky="e")
    tk.Entry(frm, textvariable=age_var, width=28).grid(row=1, column=1)

    tk.Label(frm, text="Date of Birth (DD/MM/YYYY):", bg="#e8f6ff").grid(row=2, column=0, pady=8, sticky="e")
    tk.Entry(frm, textvariable=dob_var, width=28).grid(row=2, column=1)

    def go_next():
        name = name_var.get().strip()
        age = age_var.get().strip()
        dob = dob_var.get().strip()
        if not name or not age or not dob:
            messagebox.showerror("Missing Data", "Please fill all fields before proceeding.")
            return
        try:
            int(age)
        except Exception:
            messagebox.showerror("Invalid Age", "Age must be numeric.")
            return

        front.destroy()
        open_prediction_window(name, age, dob)

    tk.Button(front, text="Next →", bg="#16a085", fg="white", font=("Arial", 12), width=12, command=go_next).pack(pady=16)

    front.mainloop()


# -------------------------------
# Start the app
# -------------------------------
if __name__ == "__main__":
    open_front_page()
