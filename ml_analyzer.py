import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

# ─── Color Palette ───────────────────────────────────────────────────────────
BG        = "#0f1117"
SURFACE   = "#1a1d27"
PANEL     = "#22263a"
ACCENT    = "#6c63ff"
ACCENT2   = "#00d4aa"
TEXT      = "#e8eaf6"
SUBTEXT   = "#8892b0"
BORDER    = "#2d3154"
SUCCESS   = "#4caf50"
WARNING   = "#ff9800"
DANGER    = "#f44336"
FONT_MAIN = ("Consolas", 10)
FONT_HEAD = ("Consolas", 13, "bold")
FONT_TITLE= ("Consolas", 18, "bold")

class MLAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ML Analyzer ◈ v2.0")
        self.root.geometry("1100x720")
        self.root.configure(bg=BG)
        self.root.resizable(True, True)

        self._apply_theme()

        # State
        self.df = None
        self.feature_cols = []
        self.target_col = tk.StringVar()
        self.task = tk.StringVar(value="classification")
        self.algorithm = tk.StringVar()
        self.label_encoders = {}
        self.model = None
        self.random_data = None

        # Build UI
        self._build_header()
        self._build_notebook()

    # ─── Theme ───────────────────────────────────────────────────────────────
    def _apply_theme(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure("TNotebook", background=BG, borderwidth=0, tabmargins=0)
        style.configure("TNotebook.Tab",
                        background=SURFACE, foreground=SUBTEXT,
                        font=("Consolas", 10, "bold"),
                        padding=[20, 8], borderwidth=0)
        style.map("TNotebook.Tab",
                  background=[("selected", PANEL)],
                  foreground=[("selected", ACCENT)])

        style.configure("TFrame", background=BG)
        style.configure("TLabelframe", background=PANEL, bordercolor=BORDER,
                        relief="flat", borderwidth=1)
        style.configure("TLabelframe.Label", background=PANEL, foreground=ACCENT,
                        font=("Consolas", 9, "bold"))

        style.configure("TCombobox",
                        fieldbackground=SURFACE, background=SURFACE,
                        foreground=TEXT, selectbackground=ACCENT,
                        arrowcolor=ACCENT, bordercolor=BORDER)
        style.map("TCombobox", fieldbackground=[("readonly", SURFACE)])

        style.configure("Vertical.TScrollbar",
                        background=PANEL, troughcolor=SURFACE,
                        arrowcolor=ACCENT, bordercolor=BORDER)
        style.configure("TRadiobutton",
                        background=PANEL, foreground=TEXT,
                        font=FONT_MAIN, focuscolor="")
        style.map("TRadiobutton", background=[("active", PANEL)])

        self.root.option_add("*TCombobox*Listbox.background", SURFACE)
        self.root.option_add("*TCombobox*Listbox.foreground", TEXT)
        self.root.option_add("*TCombobox*Listbox.selectBackground", ACCENT)

    # ─── Header ──────────────────────────────────────────────────────────────
    def _build_header(self):
        header = tk.Frame(self.root, bg=SURFACE, height=56)
        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(header, text="◈ ML ANALYZER",
                 font=("Consolas", 16, "bold"),
                 bg=SURFACE, fg=ACCENT).pack(side="left", padx=24, pady=12)

        tk.Label(header, text="Machine Learning Studio",
                 font=("Consolas", 9), bg=SURFACE, fg=SUBTEXT).pack(side="left", pady=14)

        # Status bar right side
        self.status_var = tk.StringVar(value="● Ready")
        tk.Label(header, textvariable=self.status_var,
                 font=("Consolas", 9), bg=SURFACE, fg=ACCENT2).pack(side="right", padx=24)

    # ─── Notebook ────────────────────────────────────────────────────────────
    def _build_notebook(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=0, pady=0)

        self.page1 = tk.Frame(self.notebook, bg=BG)
        self.page2 = tk.Frame(self.notebook, bg=BG)
        self.page3 = tk.Frame(self.notebook, bg=BG)

        self.notebook.add(self.page1, text="  ◉  Welcome  ")
        self.notebook.add(self.page2, text="  ◉  Dataset & Config  ")
        self.notebook.add(self.page3, text="  ◉  Model & Evaluation  ")

        self._setup_welcome()
        self._setup_dataset()
        self._setup_model()

    # ─── Helpers ─────────────────────────────────────────────────────────────
    def _btn(self, parent, text, command, color=ACCENT, width=None):
        cfg = dict(text=text, command=command, bg=color, fg="white",
                   font=("Consolas", 10, "bold"), relief="flat",
                   activebackground=ACCENT2, activeforeground="white",
                   cursor="hand2", pady=6, padx=14, bd=0)
        if width:
            cfg["width"] = width
        b = tk.Button(parent, **cfg)
        b.bind("<Enter>", lambda e: b.config(bg=ACCENT2))
        b.bind("<Leave>", lambda e: b.config(bg=color))
        return b

    def _label(self, parent, text, font=FONT_MAIN, fg=TEXT):
        return tk.Label(parent, text=text, font=font, bg=parent["bg"], fg=fg)

    def _panel(self, parent, title="", padx=10, pady=6):
        f = tk.LabelFrame(parent, text=f"  {title}  " if title else "",
                          bg=PANEL, fg=ACCENT,
                          font=("Consolas", 9, "bold"),
                          relief="flat", bd=1,
                          highlightbackground=BORDER,
                          highlightthickness=1)
        f.pack(fill="x", padx=padx, pady=pady)
        return f

    def _text_box(self, parent, height=8):
        t = scrolledtext.ScrolledText(
            parent, height=height,
            bg=SURFACE, fg=TEXT, insertbackground=ACCENT,
            font=("Consolas", 9), relief="flat", bd=0,
            selectbackground=ACCENT)
        return t

    def _set_status(self, msg, color=ACCENT2):
        self.status_var.set(f"● {msg}")

    # ─── Page 1: Welcome ─────────────────────────────────────────────────────
    def _setup_welcome(self):
        p = self.page1
        p.configure(bg=BG)

        # Center container
        center = tk.Frame(p, bg=BG)
        center.place(relx=0.5, rely=0.5, anchor="center")

        # Logo block
        logo = tk.Frame(center, bg=SURFACE, bd=0,
                        highlightbackground=ACCENT,
                        highlightthickness=2)
        logo.pack(pady=(0, 30), ipadx=40, ipady=30)

        tk.Label(logo, text="◈", font=("Consolas", 48), bg=SURFACE, fg=ACCENT).pack()
        tk.Label(logo, text="ML ANALYZER", font=("Consolas", 22, "bold"),
                 bg=SURFACE, fg=TEXT).pack()
        tk.Label(logo, text="Predict · Classify · Cluster", font=("Consolas", 10),
                 bg=SURFACE, fg=SUBTEXT).pack(pady=(4, 0))

        # Features
        features_frame = tk.Frame(center, bg=BG)
        features_frame.pack(pady=(0, 30))

        features = [
            ("🔷", "Classification", "KNN · SVM · Decision Tree"),
            ("🔶", "Regression", "Linear Regression"),
            ("🔹", "Clustering", "KMeans"),
        ]
        for icon, title, desc in features:
            row = tk.Frame(features_frame, bg=PANEL,
                           highlightbackground=BORDER, highlightthickness=1)
            row.pack(fill="x", pady=3, ipadx=16, ipady=8)
            tk.Label(row, text=icon, font=("Consolas", 14), bg=PANEL).pack(side="left", padx=(12, 8))
            tk.Label(row, text=title, font=("Consolas", 10, "bold"),
                     bg=PANEL, fg=TEXT).pack(side="left")
            tk.Label(row, text=f"  —  {desc}", font=("Consolas", 9),
                     bg=PANEL, fg=SUBTEXT).pack(side="left")

        btn = self._btn(center, "  GET STARTED  →  ",
                        lambda: self.notebook.select(1), width=26)
        btn.pack()

    # ─── Page 2: Dataset ─────────────────────────────────────────────────────
    def _setup_dataset(self):
        p = self.page2

        # Top row: Upload + columns side by side
        top = tk.Frame(p, bg=BG)
        top.pack(fill="both", expand=True, padx=10, pady=(10, 0))

        # LEFT: upload + config
        left = tk.Frame(top, bg=BG)
        left.pack(side="left", fill="both", expand=True, padx=(0, 5))

        # Upload
        uf = tk.LabelFrame(left, text="  ◈ File Upload  ",
                           bg=PANEL, fg=ACCENT, font=("Consolas", 9, "bold"),
                           relief="flat", highlightbackground=BORDER, highlightthickness=1)
        uf.pack(fill="x", pady=(0, 6))

        row = tk.Frame(uf, bg=PANEL)
        row.pack(fill="x", padx=10, pady=8)
        self.file_path_var = tk.StringVar(value="No file selected...")
        tk.Label(row, textvariable=self.file_path_var,
                 font=("Consolas", 8), bg=PANEL, fg=SUBTEXT,
                 width=38, anchor="w").pack(side="left", padx=(0, 8))
        self._btn(row, "Browse CSV", self.upload_csv).pack(side="left")

        # Target col
        tf = tk.LabelFrame(left, text="  ◈ Target Column  ",
                           bg=PANEL, fg=ACCENT, font=("Consolas", 9, "bold"),
                           relief="flat", highlightbackground=BORDER, highlightthickness=1)
        tf.pack(fill="x", pady=(0, 6))
        row2 = tk.Frame(tf, bg=PANEL)
        row2.pack(fill="x", padx=10, pady=8)
        tk.Label(row2, text="Target :", font=("Consolas", 9),
                 bg=PANEL, fg=SUBTEXT).pack(side="left")
        self.target_combobox = ttk.Combobox(row2, textvariable=self.target_col,
                                             width=24, state="readonly")
        self.target_combobox.pack(side="left", padx=8)

        # Task
        taskf = tk.LabelFrame(left, text="  ◈ Task  ",
                              bg=PANEL, fg=ACCENT, font=("Consolas", 9, "bold"),
                              relief="flat", highlightbackground=BORDER, highlightthickness=1)
        taskf.pack(fill="x", pady=(0, 6))
        row3 = tk.Frame(taskf, bg=PANEL)
        row3.pack(fill="x", padx=10, pady=8)
        for val, label in [("classification", "Classification"),
                           ("regression", "Regression"),
                           ("clustering", "Clustering")]:
            rb = tk.Radiobutton(row3, text=label, variable=self.task, value=val,
                                command=self.update_algorithm_dropdown,
                                bg=PANEL, fg=TEXT, activebackground=PANEL,
                                selectcolor=ACCENT, font=("Consolas", 9),
                                indicatoron=True)
            rb.pack(side="left", padx=10)

        # Algorithm
        af = tk.LabelFrame(left, text="  ◈ Algorithm  ",
                           bg=PANEL, fg=ACCENT, font=("Consolas", 9, "bold"),
                           relief="flat", highlightbackground=BORDER, highlightthickness=1)
        af.pack(fill="x", pady=(0, 6))
        row4 = tk.Frame(af, bg=PANEL)
        row4.pack(fill="x", padx=10, pady=8)
        tk.Label(row4, text="Algorithm :", font=("Consolas", 9),
                 bg=PANEL, fg=SUBTEXT).pack(side="left")
        self.algorithm_combobox = ttk.Combobox(row4, textvariable=self.algorithm,
                                                width=22, state="readonly")
        self.algorithm_combobox.pack(side="left", padx=8)
        self.update_algorithm_dropdown()

        # Continue btn
        self._btn(left, "  Continue to Model  →  ",
                  self.continue_to_model, color=ACCENT2).pack(pady=8, anchor="w")

        # RIGHT: columns + preview
        right = tk.Frame(top, bg=BG)
        right.pack(side="right", fill="both", expand=True, padx=(5, 0))

        cf = tk.LabelFrame(right, text="  ◈ Dataset Columns  ",
                           bg=PANEL, fg=ACCENT, font=("Consolas", 9, "bold"),
                           relief="flat", highlightbackground=BORDER, highlightthickness=1)
        cf.pack(fill="x", pady=(0, 6))
        lb_frame = tk.Frame(cf, bg=PANEL)
        lb_frame.pack(fill="both", padx=10, pady=6)
        self.columns_listbox = tk.Listbox(lb_frame, height=5,
                                           bg=SURFACE, fg=TEXT,
                                           font=("Consolas", 9),
                                           selectbackground=ACCENT,
                                           relief="flat", bd=0,
                                           activestyle="none")
        sb = tk.Scrollbar(lb_frame, orient="vertical",
                          command=self.columns_listbox.yview,
                          bg=PANEL, troughcolor=SURFACE)
        self.columns_listbox.config(yscrollcommand=sb.set)
        self.columns_listbox.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        pf = tk.LabelFrame(right, text="  ◈ Data Preview  ",
                           bg=PANEL, fg=ACCENT, font=("Consolas", 9, "bold"),
                           relief="flat", highlightbackground=BORDER, highlightthickness=1)
        pf.pack(fill="both", expand=True, pady=(0, 6))
        self.preview_text = self._text_box(pf, height=10)
        self.preview_text.pack(fill="both", expand=True, padx=8, pady=6)

    # ─── Page 3: Model ───────────────────────────────────────────────────────
    def _setup_model(self):
        p = self.page3

        # Left pane
        left = tk.Frame(p, bg=BG, width=320)
        left.pack(side="left", fill="y", padx=(10, 5), pady=10)
        left.pack_propagate(False)

        # Train
        tf = tk.LabelFrame(left, text="  ◈ Train Model  ",
                           bg=PANEL, fg=ACCENT, font=("Consolas", 9, "bold"),
                           relief="flat", highlightbackground=BORDER, highlightthickness=1)
        tf.pack(fill="x", pady=(0, 8))
        self._btn(tf, "  ▶  Train Model  ",
                  self.train_model, color=ACCENT).pack(padx=10, pady=10)

        # Random data
        rf = tk.LabelFrame(left, text="  ◈ Random Sample  ",
                           bg=PANEL, fg=ACCENT, font=("Consolas", 9, "bold"),
                           relief="flat", highlightbackground=BORDER, highlightthickness=1)
        rf.pack(fill="both", expand=True, pady=(0, 8))
        self.random_data_text = self._text_box(rf, height=10)
        self.random_data_text.pack(fill="both", expand=True, padx=8, pady=6)
        self._btn(rf, "  ↺  Generate Sample  ",
                  self.generate_random_data, color=SURFACE).pack(padx=10, pady=(0, 8))

        # Predict
        pf = tk.LabelFrame(left, text="  ◈ Predict  ",
                           bg=PANEL, fg=ACCENT, font=("Consolas", 9, "bold"),
                           relief="flat", highlightbackground=BORDER, highlightthickness=1)
        pf.pack(fill="x", pady=(0, 8))
        self._btn(pf, "  ⚡  Predict  ",
                  self.predict, color=ACCENT2).pack(padx=10, pady=10)

        # Right pane
        right = tk.Frame(p, bg=BG)
        right.pack(side="right", fill="both", expand=True, padx=(5, 10), pady=10)

        # Output
        of = tk.LabelFrame(right, text="  ◈ Evaluation Output  ",
                           bg=PANEL, fg=ACCENT, font=("Consolas", 9, "bold"),
                           relief="flat", highlightbackground=BORDER, highlightthickness=1)
        of.pack(fill="both", expand=True, pady=(0, 8))
        self.output_text = self._text_box(of, height=14)
        self.output_text.pack(fill="both", expand=True, padx=8, pady=6)

        # Plot frame
        self.plot_frame = tk.Frame(right, bg=PANEL,
                                   highlightbackground=BORDER, highlightthickness=1)
        self.plot_frame.pack(fill="both", expand=True)

    # ─── Logic ───────────────────────────────────────────────────────────────
    def upload_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if path:
            self.file_path_var.set(path.split("/")[-1])
            try:
                self.df = pd.read_csv(path)
                self._update_columns()
                self._update_preview()
                self._set_status(f"Loaded: {len(self.df)} rows × {len(self.df.columns)} cols")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load: {e}")

    def _update_columns(self):
        if self.df is not None:
            self.columns_listbox.delete(0, tk.END)
            for col in self.df.columns:
                dtype = str(self.df[col].dtype)
                self.columns_listbox.insert(tk.END, f"  {col}  [{dtype}]")
            self.target_combobox["values"] = list(self.df.columns)

    def _update_preview(self):
        if self.df is not None:
            self.preview_text.delete(1.0, tk.END)
            self.preview_text.insert(tk.END, self.df.head().to_string())

    def update_algorithm_dropdown(self):
        task = self.task.get()
        opts = {"classification": ["KNN", "SVM", "Decision Tree"],
                "regression": ["Linear Regression"],
                "clustering": ["KMeans"]}
        self.algorithm_combobox["values"] = opts[task]
        self.algorithm_combobox.current(0)

    def continue_to_model(self):
        if self.df is None:
            messagebox.showerror("Error", "Please upload a CSV file first.")
            return
        if not self.target_col.get() and self.task.get() != "clustering":
            messagebox.showerror("Error", "Please select a target column.")
            return
        self.preprocess_data()
        if self.task.get() != "clustering":
            self.feature_cols = [c for c in self.df.columns if c != self.target_col.get()]
        else:
            self.feature_cols = list(self.df.columns)
        self.notebook.select(2)
        self._set_status("Ready to train")

    def preprocess_data(self):
        if self.df is None:
            return
        self.label_encoders = {}
        for col in self.df.columns:
            if self.df[col].dtype in ["int64", "float64"]:
                self.df[col] = self.df[col].fillna(self.df[col].mean())
            else:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
        for col in self.df.columns:
            if self.df[col].dtype == "object":
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col])
                self.label_encoders[col] = le

    def generate_random_data(self):
        if self.df is None or not self.feature_cols:
            messagebox.showerror("Error", "Please upload CSV and configure first.")
            return
        try:
            self.random_data = {}
            self.random_data_text.delete(1.0, tk.END)
            self.random_data_text.insert(tk.END, "─── Generated Sample ───\n\n")
            for feat in self.feature_cols:
                if feat in self.df.columns and pd.api.types.is_numeric_dtype(self.df[feat]):
                    mn, mx = self.df[feat].min(), self.df[feat].max()
                    val = (np.random.randint(int(mn), int(mx)+1)
                           if pd.api.types.is_integer_dtype(self.df[feat])
                           else round(float(np.random.uniform(mn, mx)), 4))
                    self.random_data[feat] = val
                    self.random_data_text.insert(tk.END, f"  {feat}: {val}\n")
        except Exception as e:
            messagebox.showerror("Error", f"Error: {e}")

    def train_model(self):
        if self.df is None:
            messagebox.showerror("Error", "Please upload CSV first.")
            return
        task, algo = self.task.get(), self.algorithm.get()
        try:
            self.output_text.delete(1.0, tk.END)
            for w in self.plot_frame.winfo_children():
                w.destroy()
            self._set_status("Training...")
            self.root.update()
            if task == "classification":
                self._train_classification(algo)
            elif task == "regression":
                self._train_regression()
            else:
                self._train_clustering()
            self.generate_random_data()
            self._set_status("Training complete ✓")
        except Exception as e:
            messagebox.showerror("Error", f"Training error: {e}")
            import traceback; traceback.print_exc()

    def _write(self, text, tag=None):
        self.output_text.insert(tk.END, text)

    def _train_classification(self, algo):
        X = self.df[self.feature_cols]
        y = self.df[self.target_col.get()]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {"KNN": KNeighborsClassifier(n_neighbors=3),
                  "SVM": svm.SVC(kernel="rbf"),
                  "Decision Tree": DecisionTreeClassifier(criterion="entropy", max_depth=3)}
        self.model = models[algo]
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        self._write(f"╔══ Classification Report ══╗\n\n")
        self._write(f"  Algorithm  : {algo}\n")
        self._write(f"  Accuracy   : {acc:.4f}  ({acc*100:.1f}%)\n")
        self._write(f"  Precision  : {prec:.4f}\n")
        self._write(f"  Recall     : {rec:.4f}\n")
        self._write(f"  F1 Score   : {f1:.4f}\n\n")
        self._write(f"  Confusion Matrix:\n{cm}\n")

        # Plot
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(4, 3.2), facecolor="#22263a")
        ax.set_facecolor("#1a1d27")
        im = ax.imshow(cm, cmap="YlOrRd", interpolation="nearest")
        plt.colorbar(im, ax=ax)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white", fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted", color="#8892b0")
        ax.set_ylabel("Actual", color="#8892b0")
        ax.set_title("Confusion Matrix", color="#6c63ff", fontweight="bold")
        ax.tick_params(colors="#8892b0")
        fig.tight_layout()
        self._embed_plot(fig)

    def _train_regression(self):
        X = self.df[self.feature_cols]
        y = self.df[self.target_col.get()]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        self._write(f"╔══ Regression Report ══╗\n\n")
        self._write(f"  MAE    : {mae:.4f}\n")
        self._write(f"  RMSE   : {np.sqrt(mse):.4f}\n")
        self._write(f"  R²     : {r2:.4f}\n\n")
        df_cmp = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
        self._write(f"  Sample comparison:\n{df_cmp.head().to_string(index=False)}\n")

        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(4, 3.2), facecolor="#22263a")
        ax.set_facecolor("#1a1d27")
        ax.scatter(y_test, y_pred, alpha=0.6, color="#6c63ff", edgecolors="#00d4aa", linewidths=0.5)
        mn, mx = float(y_test.min()), float(y_test.max())
        ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect fit")
        ax.set_xlabel("Actual", color="#8892b0")
        ax.set_ylabel("Predicted", color="#8892b0")
        ax.set_title("Actual vs Predicted", color="#6c63ff", fontweight="bold")
        ax.tick_params(colors="#8892b0")
        ax.legend(facecolor="#1a1d27", edgecolor="#2d3154", labelcolor="white")
        fig.tight_layout()
        self._embed_plot(fig)

    def _train_clustering(self):
        num_cols = self.df.select_dtypes(include=["int64", "float64"]).columns[:2]
        if len(num_cols) < 2:
            messagebox.showerror("Error", "Need at least 2 numeric columns.")
            return
        X = self.df[num_cols]
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X)
        self.model = kmeans

        self._write(f"╔══ KMeans Clustering ══╗\n\n")
        self._write(f"  Clusters   : 3\n")
        self._write(f"  Inertia    : {kmeans.inertia_:.2f}\n\n")
        unique, counts = np.unique(clusters, return_counts=True)
        for c, n in zip(unique, counts):
            self._write(f"  Cluster {c}  : {n} samples\n")
        self._write(f"\n  Centroids:\n")
        for i, center in enumerate(kmeans.cluster_centers_):
            self._write(f"  [{i}]  {np.round(center, 3)}\n")

        plt.style.use("dark_background")
        colors = ["#6c63ff", "#00d4aa", "#ff9800"]
        fig, ax = plt.subplots(figsize=(4, 3.2), facecolor="#22263a")
        ax.set_facecolor("#1a1d27")
        for i in range(3):
            pts = X[clusters == i]
            ax.scatter(pts.iloc[:, 0], pts.iloc[:, 1],
                       color=colors[i], alpha=0.7, label=f"Cluster {i}", s=30)
        ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                   s=220, marker="*", c="white", zorder=5, label="Centroids",
                   edgecolors="#ff9800", linewidths=1)
        ax.set_xlabel(num_cols[0], color="#8892b0")
        ax.set_ylabel(num_cols[1], color="#8892b0")
        ax.set_title("KMeans Clusters", color="#6c63ff", fontweight="bold")
        ax.tick_params(colors="#8892b0")
        ax.legend(facecolor="#1a1d27", edgecolor="#2d3154", labelcolor="white", fontsize=8)
        fig.tight_layout()
        self._embed_plot(fig)

    def _embed_plot(self, fig):
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def predict(self):
        if self.model is None:
            messagebox.showerror("Error", "Please train a model first.")
            return
        if not self.random_data:
            messagebox.showerror("Error", "Please generate a sample first.")
            return
        try:
            input_df = pd.DataFrame([self.random_data])
            task = self.task.get()
            if task == "classification":
                pred = self.model.predict(input_df)[0]
                result = "Yes ✓" if pred in [1, True] else "No ✗"
                self._write(f"\n┌─ Prediction ─────────────\n")
                self._write(f"│  Result : {result}\n")
                self._write(f"│  Raw    : {pred}\n")
                self._write(f"└──────────────────────────\n")
            elif task == "regression":
                pred = self.model.predict(input_df)[0]
                self._write(f"\n┌─ Prediction ─────────────\n")
                self._write(f"│  Value  : {pred:.4f}\n")
                self._write(f"└──────────────────────────\n")
            else:
                cluster = self.model.predict(input_df)[0]
                self._write(f"\n┌─ Prediction ─────────────\n")
                self._write(f"│  Cluster: {cluster}\n")
                self._write(f"└──────────────────────────\n")
            self._set_status("Prediction done ✓")
        except Exception as e:
            messagebox.showerror("Error", f"Prediction error: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = MLAnalyzerApp(root)
    root.mainloop()
