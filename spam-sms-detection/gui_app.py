import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import customtkinter as ctk

# Ensure script directory is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import classifier

# Set appearance and default color theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

SAMPLE_MESSAGES = {
    "Select a pre-made sample...": "",
    "🚨 Spam: Lottery / Prize Claim": "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt CLAIM to 81010 to receive reward now!",
    "🚨 Spam: Urgent Account / Phishing": "Your bank account has been suspended due to suspicious activity. Call 0800-123-456 immediately or visit http://secure-login.com to verify.",
    "🚨 Spam: Ringtone / Free Offer": "Free ringtone service activated! Reply YES to 44222 to get unlimited polyphonic tones every week. Standard network rates apply.",
    "✅ Legitimate: Dinner & Meeting": "Hey Sarah, are we still meeting up for dinner tonight around 7:30 PM at the Italian place?",
    "✅ Legitimate: Work Follow-up": "Hi team, please find attached the revised quarterly report. Let me know if you have any feedback before tomorrow's meeting.",
    "✅ Legitimate: Delivery Notification": "Your parcel #GB-4921 has arrived and is waiting at the building reception desk. Have a great day!"
}

class SpamDetectorGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("AI Spam & Phishing Detector — Email & SMS")
        self.geometry("980x740")
        self.minsize(880, 620)

        # Center window on screen
        self.center_window(980, 740)

        # Batch results storage
        self.batch_data = []

        self.setup_ui()

    def center_window(self, width, height):
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = max(0, (screen_width - width) // 2)
        y = max(0, (screen_height - height) // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")

    def setup_ui(self):
        # 1. Header Frame
        header = ctk.CTkFrame(self, corner_radius=12, fg_color=("gray90", "#1E222A"))
        header.pack(fill="x", padx=16, pady=(14, 8))

        header_title = ctk.CTkLabel(
            header,
            text="🛡️ AI Spam & Phishing Shield",
            font=ctk.CTkFont(size=22, weight="bold")
        )
        header_title.pack(anchor="w", padx=18, pady=(12, 2))

        header_sub = ctk.CTkLabel(
            header,
            text="Fast, intelligent Email & SMS classification powered by Multinomial Naive Bayes & TF-IDF",
            font=ctk.CTkFont(size=13),
            text_color=("gray40", "gray70")
        )
        header_sub.pack(anchor="w", padx=18, pady=(0, 12))

        # Appearance mode selector in top right
        theme_frame = ctk.CTkFrame(header, fg_color="transparent")
        theme_frame.place(relx=0.98, rely=0.5, anchor="e")
        ctk.CTkLabel(theme_frame, text="Theme:", font=ctk.CTkFont(size=12)).pack(side="left", padx=5)
        self.theme_menu = ctk.CTkOptionMenu(
            theme_frame,
            values=["Dark", "Light", "System"],
            command=self.change_theme,
            width=90,
            height=28
        )
        self.theme_menu.set("Dark")
        self.theme_menu.pack(side="left")

        # 2. Main Tabview
        self.tabview = ctk.CTkTabview(self, corner_radius=12)
        self.tabview.pack(fill="both", expand=True, padx=16, pady=8)

        self.tab_single = self.tabview.add("✉️ Quick Detector")
        self.tab_batch = self.tabview.add("📁 Batch File Analyzer")
        self.tab_insights = self.tabview.add("📊 Model & Insights")

        self.setup_single_tab()
        self.setup_batch_tab()
        self.setup_insights_tab()

        # 3. Status Bar
        self.status_bar = ctk.CTkLabel(
            self,
            text="Ready • Model loaded (MultinomialNB + TF-IDF) • Accuracy: ~96% | Precision: 100%",
            font=ctk.CTkFont(size=11),
            text_color="gray60",
            anchor="w"
        )
        self.status_bar.pack(fill="x", padx=20, pady=(2, 6))

    def change_theme(self, choice):
        ctk.set_appearance_mode(choice)

    # ========================================================
    # TAB 1: QUICK DETECTOR
    # ========================================================
    def setup_single_tab(self):
        tab = self.tab_single

        # Sample message picker
        top_bar = ctk.CTkFrame(tab, fg_color="transparent")
        top_bar.pack(fill="x", padx=10, pady=(6, 4))

        ctk.CTkLabel(top_bar, text="Quick Test Samples:", font=ctk.CTkFont(size=12, weight="bold")).pack(side="left", padx=(0, 8))
        self.sample_menu = ctk.CTkOptionMenu(
            top_bar,
            values=list(SAMPLE_MESSAGES.keys()),
            command=self.load_sample,
            width=320,
            height=30
        )
        self.sample_menu.pack(side="left")

        # Paste button
        paste_btn = ctk.CTkButton(
            top_bar,
            text="📋 Paste",
            width=80,
            height=30,
            command=self.paste_clipboard,
            fg_color=("gray75", "#2C313C"),
            hover_color=("gray65", "#3C4250")
        )
        paste_btn.pack(side="right", padx=(6, 0))

        # Clear button
        clear_btn = ctk.CTkButton(
            top_bar,
            text="🧹 Clear",
            width=80,
            height=30,
            command=self.clear_input,
            fg_color=("gray75", "#2C313C"),
            hover_color=("gray65", "#3C4250")
        )
        clear_btn.pack(side="right")

        # Input Text Box
        self.input_textbox = ctk.CTkTextbox(
            tab,
            height=130,
            font=ctk.CTkFont(size=13),
            corner_radius=10,
            wrap="word"
        )
        self.input_textbox.pack(fill="x", padx=10, pady=6)
        self.input_textbox.bind("<KeyRelease>", self.update_char_count)

        # Counter & Analyze Row
        action_row = ctk.CTkFrame(tab, fg_color="transparent")
        action_row.pack(fill="x", padx=10, pady=(2, 10))

        self.char_count_lbl = ctk.CTkLabel(
            action_row,
            text="0 characters • 0 words",
            font=ctk.CTkFont(size=11),
            text_color="gray60"
        )
        self.char_count_lbl.pack(side="left")

        self.analyze_btn = ctk.CTkButton(
            action_row,
            text="🔍 Analyze Message",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=38,
            width=180,
            command=self.run_classification
        )
        self.analyze_btn.pack(side="right")

        # Results Container
        self.result_container = ctk.CTkFrame(tab, corner_radius=12, fg_color=("gray90", "#1C2026"))
        self.result_container.pack(fill="both", expand=True, padx=10, pady=(0, 6))

        # Result Banner
        self.banner_frame = ctk.CTkFrame(
            self.result_container,
            corner_radius=10,
            fg_color=("gray85", "#262C36"),
            height=54
        )
        self.banner_frame.pack(fill="x", padx=14, pady=12)

        self.banner_label = ctk.CTkLabel(
            self.banner_frame,
            text="Ready to analyze — Enter or paste an SMS / Email message above",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.banner_label.pack(pady=12)

        # Metrics Card Frame
        self.metrics_frame = ctk.CTkFrame(self.result_container, fg_color="transparent")
        self.metrics_frame.pack(fill="x", padx=14, pady=4)

        # Spam Probability Box
        self.spam_metric_card = ctk.CTkFrame(self.metrics_frame, corner_radius=10, fg_color=("gray85", "#262C36"))
        self.spam_metric_card.pack(side="left", fill="both", expand=True, padx=(0, 6), pady=4)
        ctk.CTkLabel(self.spam_metric_card, text="Spam Probability", font=ctk.CTkFont(size=12)).pack(pady=(8, 2))
        self.spam_prob_val = ctk.CTkLabel(self.spam_metric_card, text="--%", font=ctk.CTkFont(size=22, weight="bold"))
        self.spam_prob_val.pack(pady=(0, 4))
        self.spam_progress = ctk.CTkProgressBar(self.spam_metric_card, height=10, progress_color="#E53935")
        self.spam_progress.set(0)
        self.spam_progress.pack(fill="x", padx=14, pady=(0, 10))

        # Legitimate Probability Box
        self.ham_metric_card = ctk.CTkFrame(self.metrics_frame, corner_radius=10, fg_color=("gray85", "#262C36"))
        self.ham_metric_card.pack(side="right", fill="both", expand=True, padx=(6, 0), pady=4)
        ctk.CTkLabel(self.ham_metric_card, text="Legitimate (Ham) Probability", font=ctk.CTkFont(size=12)).pack(pady=(8, 2))
        self.ham_prob_val = ctk.CTkLabel(self.ham_metric_card, text="--%", font=ctk.CTkFont(size=22, weight="bold"))
        self.ham_prob_val.pack(pady=(0, 4))
        self.ham_progress = ctk.CTkProgressBar(self.ham_metric_card, height=10, progress_color="#2E7D32")
        self.ham_progress.set(0)
        self.ham_progress.pack(fill="x", padx=14, pady=(0, 10))

        # Keyword and Token Insights
        self.diagnostics_frame = ctk.CTkFrame(self.result_container, corner_radius=10, fg_color=("gray85", "#262C36"))
        self.diagnostics_frame.pack(fill="both", expand=True, padx=14, pady=(6, 12))

        ctk.CTkLabel(
            self.diagnostics_frame,
            text="🧠 Diagnostic Breakdown & Trigger Keywords:",
            font=ctk.CTkFont(size=12, weight="bold")
        ).pack(anchor="w", padx=12, pady=(8, 2))

        self.keyword_chips_box = ctk.CTkTextbox(
            self.diagnostics_frame,
            height=65,
            font=ctk.CTkFont(size=12),
            corner_radius=8,
            state="disabled",
            wrap="word"
        )
        self.keyword_chips_box.pack(fill="both", expand=True, padx=12, pady=(2, 10))

    def update_char_count(self, event=None):
        content = self.input_textbox.get("1.0", "end-1c")
        chars = len(content)
        words = len(content.split())
        self.char_count_lbl.configure(text=f"{chars} characters • {words} words")

    def load_sample(self, choice):
        text = SAMPLE_MESSAGES.get(choice, "")
        if text:
            self.input_textbox.delete("1.0", "end")
            self.input_textbox.insert("1.0", text)
            self.update_char_count()
            self.run_classification()

    def paste_clipboard(self):
        try:
            text = self.clipboard_get()
            self.input_textbox.delete("1.0", "end")
            self.input_textbox.insert("1.0", text)
            self.update_char_count()
        except Exception:
            pass

    def clear_input(self):
        self.input_textbox.delete("1.0", "end")
        self.update_char_count()
        self.sample_menu.set("Select a pre-made sample...")
        self.banner_frame.configure(fg_color=("gray85", "#262C36"))
        self.banner_label.configure(text="Ready to analyze — Enter or paste an SMS / Email message above", text_color=("black", "white"))
        self.spam_prob_val.configure(text="--%")
        self.ham_prob_val.configure(text="--%")
        self.spam_progress.set(0)
        self.ham_progress.set(0)
        self.keyword_chips_box.configure(state="normal")
        self.keyword_chips_box.delete("1.0", "end")
        self.keyword_chips_box.configure(state="disabled")

    def run_classification(self):
        text = self.input_textbox.get("1.0", "end-1c").strip()
        if not text:
            messagebox.showwarning("Empty Message", "Please type or paste a message to analyze.")
            return

        res = classifier.classify_text(text)

        spam_p = res["spam_probability"]
        ham_p = res["ham_probability"]
        self.spam_prob_val.configure(text=f"{spam_p}%")
        self.ham_prob_val.configure(text=f"{ham_p}%")
        self.spam_progress.set(spam_p / 100.0)
        self.ham_progress.set(ham_p / 100.0)

        if res["is_spam"]:
            self.banner_frame.configure(fg_color="#8B1E1E")
            self.banner_label.configure(
                text=f"🚨 SPAM / PHISHING DETECTED ({res['confidence']}% Confidence)",
                text_color="#FFFFFF"
            )
        else:
            self.banner_frame.configure(fg_color="#1E6B38")
            self.banner_label.configure(
                text=f"✅ LEGITIMATE MESSAGE (HAM) ({res['confidence']}% Confidence)",
                text_color="#FFFFFF"
            )

        # Keywords diagnostic display
        self.keyword_chips_box.configure(state="normal")
        self.keyword_chips_box.delete("1.0", "end")

        diag_lines = [
            f"Preprocessed Stemmed Tokens: {res['transformed_text'] or '[None]'}",
            ""
        ]

        if res["flagged_keywords"]:
            diag_lines.append("Flagged High-Risk Spam Keywords:")
            for word, score in res["flagged_keywords"]:
                diag_lines.append(f"  • '{word}' (Spam Indicator Weight: +{score})")
        else:
            diag_lines.append("No high-risk spam keywords detected in message tokens.")

        self.keyword_chips_box.insert("1.0", "\n".join(diag_lines))
        self.keyword_chips_box.configure(state="disabled")

    # ========================================================
    # TAB 2: BATCH ANALYZER
    # ========================================================
    def setup_batch_tab(self):
        tab = self.tab_batch

        # Top Control Frame
        ctrl = ctk.CTkFrame(tab, fg_color="transparent")
        ctrl.pack(fill="x", padx=10, pady=8)

        load_btn = ctk.CTkButton(
            ctrl,
            text="📁 Select File (CSV / TXT)",
            command=self.load_batch_file,
            font=ctk.CTkFont(size=13, weight="bold"),
            width=180,
            height=34
        )
        load_btn.pack(side="left", padx=(0, 10))

        self.batch_file_lbl = ctk.CTkLabel(
            ctrl,
            text="No file selected (Supports .csv with text column or .txt with one message per line)",
            font=ctk.CTkFont(size=12),
            text_color="gray60"
        )
        self.batch_file_lbl.pack(side="left")

        self.export_btn = ctk.CTkButton(
            ctrl,
            text="💾 Export Results CSV",
            state="disabled",
            command=self.export_batch_results,
            width=160,
            height=34
        )
        self.export_btn.pack(side="right")

        # Summary Metrics Bar
        self.batch_summary_frame = ctk.CTkFrame(tab, corner_radius=10, fg_color=("gray90", "#1C2026"))
        self.batch_summary_frame.pack(fill="x", padx=10, pady=6)

        self.b_total_lbl = ctk.CTkLabel(self.batch_summary_frame, text="Total: 0", font=ctk.CTkFont(size=13, weight="bold"))
        self.b_total_lbl.pack(side="left", padx=16, pady=8)

        self.b_spam_lbl = ctk.CTkLabel(self.batch_summary_frame, text="Spam: 0 (0%)", text_color="#E53935", font=ctk.CTkFont(size=13, weight="bold"))
        self.b_spam_lbl.pack(side="left", padx=16, pady=8)

        self.b_ham_lbl = ctk.CTkLabel(self.batch_summary_frame, text="Legitimate: 0 (0%)", text_color="#2E7D32", font=ctk.CTkFont(size=13, weight="bold"))
        self.b_ham_lbl.pack(side="left", padx=16, pady=8)

        # Filter option
        ctk.CTkLabel(self.batch_summary_frame, text="Filter:", font=ctk.CTkFont(size=12)).pack(side="left", padx=(20, 6))
        self.filter_menu = ctk.CTkOptionMenu(
            self.batch_summary_frame,
            values=["All Messages", "Only Spam", "Only Legitimate"],
            command=self.filter_batch_table,
            width=140,
            height=26
        )
        self.filter_menu.pack(side="left")

        # Results Table (Tkinter Treeview with dark styling)
        table_frame = ctk.CTkFrame(tab, corner_radius=10, fg_color=("gray90", "#1C2026"))
        table_frame.pack(fill="both", expand=True, padx=10, pady=(6, 10))

        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Treeview",
            background="#23272E",
            foreground="#DCE1E8",
            rowheight=26,
            fieldbackground="#23272E",
            font=("Segoe UI", 10)
        )
        style.configure(
            "Treeview.Heading",
            background="#1E222A",
            foreground="#FFFFFF",
            font=("Segoe UI", 10, "bold")
        )
        style.map("Treeview", background=[("selected", "#3E4451")])

        cols = ("ID", "Label", "Confidence", "Spam Prob", "Message")
        self.tree = ttk.Treeview(table_frame, columns=cols, show="headings")
        self.tree.heading("ID", text="#")
        self.tree.heading("Label", text="Prediction")
        self.tree.heading("Confidence", text="Confidence")
        self.tree.heading("Spam Prob", text="Spam %")
        self.tree.heading("Message", text="Message Preview")

        self.tree.column("ID", width=45, anchor="center")
        self.tree.column("Label", width=95, anchor="center")
        self.tree.column("Confidence", width=95, anchor="center")
        self.tree.column("Spam Prob", width=85, anchor="center")
        self.tree.column("Message", width=550, anchor="w")

        tree_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)

        self.tree.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=8)
        tree_scroll.pack(side="right", fill="y", padx=(0, 8), pady=8)

        # Tags for colored labels
        self.tree.tag_configure("spam_tag", foreground="#FF6B6B")
        self.tree.tag_configure("ham_tag", foreground="#51CF66")

    def load_batch_file(self):
        file_path = filedialog.askopenfilename(
            title="Select CSV or TXT File",
            filetypes=[("Data Files", "*.csv *.txt"), ("CSV Files", "*.csv"), ("Text Files", "*.txt")]
        )
        if not file_path:
            return

        self.batch_file_lbl.configure(text=f"Loading: {os.path.basename(file_path)}...")

        threading.Thread(target=self._process_batch_file, args=(file_path,), daemon=True).start()

    def _process_batch_file(self, file_path):
        messages = []
        try:
            if file_path.endswith(".csv"):
                import pandas as pd
                df = pd.read_csv(file_path, encoding="latin-1")
                # Detect text column
                text_col = None
                for candidate in ["text", "v2", "message", "sms", "email", "content", "body"]:
                    if candidate in df.columns:
                        text_col = candidate
                        break
                if text_col is None:
                    # Pick first string column
                    for col in df.columns:
                        if df[col].dtype == object:
                            text_col = col
                            break
                if text_col is None:
                    text_col = df.columns[0]
                messages = df[text_col].dropna().astype(str).tolist()
            else:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    messages = [line.strip() for line in f if line.strip()]
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("File Error", f"Failed to read file: {e}"))
            return

        results = []
        for i, msg in enumerate(messages, 1):
            res = classifier.classify_text(msg)
            results.append({
                "id": i,
                "label": res["label"],
                "confidence": f"{res['confidence']}%",
                "spam_prob": f"{res['spam_probability']}%",
                "message": msg
            })

        self.batch_data = results
        self.after(0, lambda: self._update_batch_ui(file_path, results))

    def _update_batch_ui(self, file_path, results):
        self.batch_file_lbl.configure(text=f"Loaded: {os.path.basename(file_path)} ({len(results)} rows)")
        self.export_btn.configure(state="normal")

        total = len(results)
        spam_count = sum(1 for r in results if r["label"] == "Spam")
        ham_count = total - spam_count
        spam_pct = round((spam_count / total * 100), 1) if total > 0 else 0
        ham_pct = round((ham_count / total * 100), 1) if total > 0 else 0

        self.b_total_lbl.configure(text=f"Total: {total}")
        self.b_spam_lbl.configure(text=f"Spam: {spam_count} ({spam_pct}%)")
        self.b_ham_lbl.configure(text=f"Legitimate: {ham_count} ({ham_pct}%)")

        self.filter_batch_table("All Messages")

    def filter_batch_table(self, choice):
        # Clear existing rows
        for item in self.tree.get_children():
            self.tree.delete(item)

        for r in self.batch_data:
            if choice == "Only Spam" and r["label"] != "Spam":
                continue
            if choice == "Only Legitimate" and r["label"] != "Not Spam":
                continue

            tag = "spam_tag" if r["label"] == "Spam" else "ham_tag"
            msg_snippet = (r["message"][:100] + "...") if len(r["message"]) > 100 else r["message"]
            self.tree.insert("", "end", values=(r["id"], r["label"], r["confidence"], r["spam_prob"], msg_snippet), tags=(tag,))

    def export_batch_results(self):
        if not self.batch_data:
            return
        save_path = filedialog.asksaveasfilename(
            title="Save Classified Results",
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv")]
        )
        if not save_path:
            return

        try:
            import pandas as pd
            df = pd.DataFrame(self.batch_data)
            df.to_csv(save_path, index=False)
            messagebox.showinfo("Export Success", f"Successfully exported {len(self.batch_data)} records to:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to save file: {e}")

    # ========================================================
    # TAB 3: MODEL INSIGHTS
    # ========================================================
    def setup_insights_tab(self):
        tab = self.tab_insights

        # Model Info Card
        info_frame = ctk.CTkFrame(tab, corner_radius=10, fg_color=("gray90", "#1C2026"))
        info_frame.pack(fill="x", padx=10, pady=8)

        ctk.CTkLabel(info_frame, text="⚡ Model Architecture & Performance Specs", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=14, pady=(10, 4))

        stats_text = (
            "• Primary Classifier: Multinomial Naive Bayes (MultinomialNB)\n"
            "• Vectorizer: Term Frequency - Inverse Document Frequency (TF-IDF)\n"
            "• Test Accuracy: 95.94% on held-out validation set\n"
            "• Test Precision: 100.0% (Zero false positives in testing)\n"
            "• Training Dataset: 5,169 SMS & Email messages (SMS Spam Collection Dataset)\n"
            "• Preprocessing Pipeline: Lowercase -> Tokenize -> Filter Alpha-numeric -> Stopword & Punctuation removal -> Porter Stemmer"
        )
        ctk.CTkLabel(info_frame, text=stats_text, justify="left", font=ctk.CTkFont(size=12), text_color=("gray20", "gray80")).pack(anchor="w", padx=14, pady=(0, 12))

        # Top Spam Trigger Keywords Box
        kw_frame = ctk.CTkFrame(tab, corner_radius=10, fg_color=("gray90", "#1C2026"))
        kw_frame.pack(fill="both", expand=True, padx=10, pady=(4, 10))

        ctk.CTkLabel(kw_frame, text="🔥 Top 25 AI Spam Indicator Keywords (Highest Log-Odds Weight)", font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w", padx=14, pady=(10, 6))

        top_keywords = [
            "1. claim (+3.48)", "2. prize (+3.39)", "3. tone (+3.22)", "4. guarante (+3.05)",
            "5. award (+2.98)", "6. servic (+2.84)", "7. txt (+2.78)", "8. rington (+2.71)",
            "9. nokia (+2.65)", "10. valid (+2.52)", "11. code (+2.48)", "12. video (+2.42)",
            "13. http (+2.38)", "14. mobil (+2.29)", "15. urgent (+2.22)", "16. cash (+2.18)",
            "17. free (+2.12)", "18. winner (+1.89)", "19. text (+1.85)", "20. custom (+1.79)",
            "21. contact (+1.75)", "22. call (+1.72)", "23. stop (+1.68)", "24. repli (+1.65)", "25. win (+1.61)"
        ]

        grid_frame = ctk.CTkFrame(kw_frame, fg_color="transparent")
        grid_frame.pack(fill="both", expand=True, padx=14, pady=(0, 12))

        for idx, kw in enumerate(top_keywords):
            row = idx % 9
            col = idx // 9
            chip = ctk.CTkLabel(
                grid_frame,
                text=kw,
                font=ctk.CTkFont(size=11),
                fg_color=("gray80", "#262C36"),
                corner_radius=6,
                padx=8,
                pady=4
            )
            chip.grid(row=row, column=col, padx=4, pady=3, sticky="w")


if __name__ == "__main__":
    app = SpamDetectorGUI()
    app.mainloop()
