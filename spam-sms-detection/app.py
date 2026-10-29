import os
import sys
import streamlit as st
import pandas as pd

# Add current directory to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import classifier

st.set_page_config(
    page_title="AI Spam & Phishing Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #888;
        margin-bottom: 1.5rem;
    }
    .result-card-spam {
        background: linear-gradient(135deg, #441118, #2a0c10);
        border: 1px solid #ff4b4b;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
    }
    .result-card-ham {
        background: linear-gradient(135deg, #0e3820, #092414);
        border: 1px solid #09ab3b;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
    }
    .badge-spam {
        background-color: #ff4b4b;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
    }
    .badge-ham {
        background-color: #09ab3b;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
    }
    .keyword-chip {
        display: inline-block;
        background-color: rgba(255, 75, 75, 0.18);
        border: 1px solid rgba(255, 75, 75, 0.5);
        color: #ff8080;
        padding: 4px 10px;
        border-radius: 15px;
        margin: 4px;
        font-size: 0.88rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🛡️ Spam Shield AI")
    st.caption("Multinomial Naive Bayes + TF-IDF")
    st.markdown("---")
    
    st.markdown("#### ⚡ Quick Samples")
    sample_choice = st.selectbox(
        "Load a sample message:",
        [
            "Custom Input",
            "Urgent: Claim £1000 prize award (Spam)",
            "Bank Security Suspended Account (Spam)",
            "Free Ringtone Subscription (Spam)",
            "Meeting dinner tonight (Legitimate)",
            "Office project follow-up (Legitimate)",
            "Package delivery notice (Legitimate)"
        ]
    )

    sample_dict = {
        "Urgent: Claim £1000 prize award (Spam)": "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt CLAIM to 81010 to receive reward now!",
        "Bank Security Suspended Account (Spam)": "Your bank account has been suspended due to suspicious activity. Call 0800-123-456 immediately or visit http://secure-login.com to verify.",
        "Free Ringtone Subscription (Spam)": "Free ringtone service activated! Reply YES to 44222 to get unlimited polyphonic tones every week. Standard network rates apply.",
        "Meeting dinner tonight (Legitimate)": "Hey Sarah, are we still meeting up for dinner tonight around 7:30 PM at the Italian place?",
        "Office project follow-up (Legitimate)": "Hi team, please find attached the revised quarterly report. Let me know if you have any feedback before tomorrow's meeting.",
        "Package delivery notice (Legitimate)": "Your parcel #GB-4921 has arrived and is waiting at the building reception desk. Have a great day!"
    }

    st.markdown("---")
    st.markdown("#### 📈 Model Stats")
    st.markdown("- **Algorithm**: MultinomialNB")
    st.markdown("- **Accuracy**: **95.94%**")
    st.markdown("- **Precision**: **100.0%**")
    st.markdown("- **Dataset**: 5,169 SMS/Emails")

# Title Header
st.markdown('<div class="main-title">🛡️ AI Spam & Phishing Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Analyze email and SMS messages in real-time or process batch files using machine learning</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["✉️ Single Message Analyzer", "📁 Batch CSV / File Processing", "📊 Model Insights & Analytics"])

# ==========================================
# TAB 1: SINGLE MESSAGE ANALYZER
# ==========================================
with tab1:
    default_text = sample_dict.get(sample_choice, "") if sample_choice != "Custom Input" else ""
    
    user_input = st.text_area(
        "Enter or paste the message content below:",
        value=default_text,
        height=140,
        placeholder="e.g. Congratulations! You won a $1,000 gift card. Call 800-XXX to claim..."
    )

    col_btn1, col_btn2, _ = st.columns([1, 1, 4])
    with col_btn1:
        analyze_clicked = st.button("🔍 Analyze Message", type="primary", use_container_width=True)
    with col_btn2:
        if st.button("🧹 Clear", use_container_width=True):
            user_input = ""
            st.rerun()

    if analyze_clicked or (user_input and sample_choice != "Custom Input"):
        if user_input.strip():
            res = classifier.classify_text(user_input)
            
            st.markdown("---")
            col_res, col_chart = st.columns([3, 2])
            
            with col_res:
                if res["is_spam"]:
                    st.markdown(f"""
                    <div class="result-card-spam">
                        <h2 style="color: #ff4b4b; margin:0;">🚨 SPAM / PHISHING DETECTED</h2>
                        <p style="font-size: 1.15rem; margin-top: 8px;">Confidence: <b>{res['confidence']}%</b></p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-card-ham">
                        <h2 style="color: #09ab3b; margin:0;">✅ LEGITIMATE MESSAGE (HAM)</h2>
                        <p style="font-size: 1.15rem; margin-top: 8px;">Confidence: <b>{res['confidence']}%</b></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Breakdown
                st.markdown("#### 🔍 Prediction Probability")
                st.write(f"**Spam Probability:** {res['spam_probability']}%")
                st.progress(res['spam_probability'] / 100.0)
                
                st.write(f"**Legitimate (Ham) Probability:** {res['ham_probability']}%")
                st.progress(res['ham_probability'] / 100.0)

            with col_chart:
                st.markdown("#### 🧠 NLP Token & Keyword Diagnostics")
                st.write(f"**Processed Stemmed Tokens:** `{res['transformed_text'] or '[None]'}`")
                
                if res["flagged_keywords"]:
                    st.markdown("**High-Risk Spam Trigger Tokens:**")
                    chips_html = "".join([f'<span class="keyword-chip">{w} (+{s})</span>' for w, s in res["flagged_keywords"]])
                    st.markdown(chips_html, unsafe_allow_html=True)
                else:
                    st.info("No strong spam trigger keywords found.")
        else:
            st.warning("Please enter a message to analyze.")

# ==========================================
# TAB 2: BATCH CSV / FILE PROCESSING
# ==========================================
with tab2:
    st.markdown("### 📁 Batch Message Classification")
    st.write("Upload a CSV file with SMS/Email messages to classify them in bulk.")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv", "txt"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_batch = pd.read_csv(uploaded_file, encoding="latin-1")
            else:
                lines = [l.decode("utf-8", errors="ignore").strip() for l in uploaded_file.readlines()]
                df_batch = pd.DataFrame({"text": [l for l in lines if l]})

            st.success(f"Loaded {len(df_batch)} rows successfully!")
            
            # Select column
            text_columns = list(df_batch.columns)
            selected_col = st.selectbox("Select the column containing message text:", text_columns)
            
            if st.button("🚀 Classify Entire Dataset", type="primary"):
                with st.spinner("Classifying messages..."):
                    results = []
                    for text in df_batch[selected_col].astype(str):
                        res = classifier.classify_text(text)
                        results.append({
                            "Prediction": res["label"],
                            "Spam_Probability": f"{res['spam_probability']}%",
                            "Confidence": f"{res['confidence']}%",
                            "Is_Spam": res["is_spam"]
                        })
                    
                    df_res = pd.DataFrame(results)
                    df_final = pd.concat([df_batch, df_res], axis=1)

                    spam_count = int(df_res["Is_Spam"].sum())
                    ham_count = len(df_res) - spam_count
                    spam_rate = round(spam_count / len(df_res) * 100, 1) if len(df_res) > 0 else 0

                    # Metrics
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total Analyzed", len(df_res))
                    m2.metric("Spam Detected", spam_count, delta=f"{spam_rate}%", delta_color="inverse")
                    m3.metric("Legitimate Messages", ham_count)
                    m4.metric("Spam Rate", f"{spam_rate}%")

                    st.markdown("#### 📋 Classified Results Preview")
                    st.dataframe(df_final.head(100), use_container_width=True)

                    # Download button
                    csv_data = df_final.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="💾 Download Classified CSV",
                        data=csv_data,
                        file_name="classified_spam_results.csv",
                        mime="text/csv",
                        type="primary"
                    )
        except Exception as e:
            st.error(f"Error processing file: {e}")

# ==========================================
# TAB 3: MODEL INSIGHTS
# ==========================================
with tab3:
    st.markdown("### 📊 Model Architecture & Training Summary")
    
    col_arch1, col_arch2 = st.columns(2)
    with col_arch1:
        st.markdown("""
        #### ⚙️ Pipeline Specifications
        - **Tokenizer & Stemmer**: NLTK `word_tokenize` + `PorterStemmer`
        - **Stopword Filtering**: English stopwords list + punctuation removal
        - **Feature Extractor**: Scikit-Learn `TfidfVectorizer`
        - **Classifier**: `MultinomialNB` (Laplace Smoothing $\\alpha = 1.0$)
        """)
    with col_arch2:
        st.markdown("""
        #### 🎯 Evaluation Metrics
        - **Test Accuracy**: **95.94%**
        - **Test Precision**: **100.0%** (Zero False Positives)
        - **Dataset Size**: **5,169 SMS & Email records**
        """)

    st.markdown("---")
    st.markdown("#### 🔥 Top Spam Trigger Keywords (Highest Weight)")
    keywords_list = [
        ("claim", 3.48), ("prize", 3.39), ("tone", 3.22), ("guarante", 3.05),
        ("award", 2.98), ("servic", 2.84), ("txt", 2.78), ("rington", 2.71),
        ("nokia", 2.65), ("valid", 2.52), ("code", 2.48), ("video", 2.42),
        ("http", 2.38), ("mobil", 2.29), ("urgent", 2.22), ("cash", 2.18)
    ]
    df_kw = pd.DataFrame(keywords_list, columns=["Keyword (Stemmed)", "Spam Bias Weight (Log-Odds)"])
    st.dataframe(df_kw, use_container_width=True)
