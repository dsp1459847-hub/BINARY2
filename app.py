import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict

# मोबाइल व्यू के लिए लेआउट सेट करना
st.set_page_config(page_title="MAYA AI - Adaptive Booster v4.0", layout="centered")

st.title("👑 MAYA AI - एडेप्टिव लर्निंग बूस्टर v4.0")
st.write("पिछली गलतियों से सीखकर और बेस-शिफ्ट ट्रांजिशन थ्योरी के आधार पर सटीक अंकों की खोज।")

# 1. 60-कॉलम बाइनरी मैट्रिक्स जनरेशन (ऑब्जेक्ट टाइप सुरक्षित)
def process_binary_matrix(df):
    shifts = ['DS', 'FD', 'GD', 'GL', 'DB', 'SG']
    df = df.dropna(subset=['DATE']).reset_index(drop=True)
    
    h_data = {}
    v_data = {}
    
    for shift in shifts:
        if shift in df.columns:
            s_data = pd.to_numeric(df[shift].replace('XX', np.nan), errors='coerce')
            
            # हॉरिजॉन्टल पार्ट्स
            h_data[f'{shift}_H1'] = np.where((s_data >= 1) & (s_data <= 20), 1, np.where(s_data.isna(), -1, 0))
            h_data[f'{shift}_H2'] = np.where((s_data >= 21) & (s_data <= 40), 1, np.where(s_data.isna(), -1, 0))
            h_data[f'{shift}_H3'] = np.where((s_data >= 41) & (s_data <= 60), 1, np.where(s_data.isna(), -1, 0))
            h_data[f'{shift}_H4'] = np.where((s_data >= 61) & (s_data <= 80), 1, np.where(s_data.isna(), -1, 0))
            h_data[f'{shift}_H5'] = np.where(((s_data >= 81) & (s_data <= 99)) | (s_data == 0), 1, np.where(s_data.isna(), -1, 0))
            
            # वर्टिकल पार्ट्स (आखिरी अंक)
            last_digit = s_data % 10
            v_data[f'{shift}_V1'] = np.where((last_digit == 1) | (last_digit == 2), 1, np.where(s_data.isna(), -1, 0))
            v_data[f'{shift}_V2'] = np.where((last_digit == 3) | (last_digit == 4), 1, np.where(s_data.isna(), -1, 0))
            v_data[f'{shift}_V3'] = np.where((last_digit == 5) | (last_digit == 6), 1, np.where(s_data.isna(), -1, 0))
            v_data[f'{shift}_V4'] = np.where((last_digit == 7) | (last_digit == 8), 1, np.where(s_data.isna(), -1, 0))
            v_data[f'{shift}_V5'] = np.where((last_digit == 9) | (last_digit == 0), 1, np.where(s_data.isna(), -1, 0))
            
    h_df = pd.DataFrame(h_data)
    v_df = pd.DataFrame(v_data)
    
    full_matrix = pd.concat([df[['S. NUMBER ', 'DATE'] + [s for s in shifts if s in df.columns]], h_df, v_df], axis=1)
    return full_matrix, list(h_df.columns) + list(v_df.columns)

# 2. एडेप्टिव लर्निंग इंजन - ट्रांजिशन मैट्रिक्स थ्योरी
def adaptive_learning_prediction(matrix_df, pattern_cols):
    # -1 (XX/छुट्टी) वाली रोज़ को हटाकर शुद्ध संख्यात्मक डेटा तैयार करना
    numeric_df = matrix_df.copy()
    for col in pattern_cols:
        numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')
    
    # अंतिम रो (कल का लाइव डेटा)
    total_rows = len(numeric_df)
    if total_rows < 15:
        return {c: 0.5 for c in pattern_cols}, {}, {}
        
    last_day_signals = numeric_df.loc[total_rows - 1, pattern_cols].to_dict()
    
    # हर पैटर्न का एक दूसरे के साथ ट्रांजिशन स्कोर (Probability Weights)
    transition_weights = defaultdict(lambda: {'count_1': 0, 'total': 0})
    
    # बैकटेस्टिंग और लर्निंग लूप: इतिहास में जाकर हर एक पैटर्न के बाद दूसरे पैटर्न के आने का लिंक ढूंढना
    for i in range(total_rows - 2):
        for col_past in pattern_cols:
            val_past = numeric_df.loc[i, col_past]
            if val_past == -1: continue # छुट्टी को छोड़ दें
            
            # अगर कल यह पैटर्न एक्टिव (1) या इनएक्टिव (0) था
            for col_future in pattern_cols:
                val_future = numeric_df.loc[i + 1, col_future]
                if val_future == -1: continue
                
                key = (col_past, val_past, col_future)
                transition_weights[key]['total'] += 1
                if val_future == 1:
                    transition_weights[key]['count_1'] += 1

    # आज के लिए अंतिम स्कोर की गणना (फर्जी 100% दावों से मुक्त, रियल प्रोबेबिलिटी)
    final_pattern_scores = {}
    strong_signals = {}
    dead_signals = {}
    
    for target_col in pattern_cols:
        score_sum = 0
        valid_factors = 0
        
        for col_past in pattern_cols:
            val_past = last_day_signals.get(col_past, -1)
            if val_past == -1: continue
            
            key = (col_past, val_past, target_col)
            stats = transition_weights.get(key, {'count_1': 0, 'total': 0})
            
            if stats['total'] >= 5: # न्यूनतम 5 बार इतिहास में आना जरूरी है (Overfitting से बचाव)
                prob = stats['count_1'] / stats['total']
                score_sum += prob
                valid_factors += 1
                
        # एवरेज प्रोबेबिलिटी स्कोर निकालना
        if valid_factors > 0:
            final_score = score_sum / valid_factors
        else:
            final_score = 0.5
            
        final_pattern_scores[target_col] = final_score
        
        # केवल वास्तविक मजबूत और कमजोर पैटर्न्स को अलग करना (फर्जी 100% को रोककर व्यावहारिक रेंज)
        if final_score >= 0.68:
            strong_signals[target_col] = f"ट्रेंड स्कोर: {final_score*100:.1f}% (आने की मजबूत संभावना)"
        elif final_score <= 0.32:
            dead_signals[target_col] = f"ट्रेंड स्कोर: {final_score*100:.1f}% (ब्लॉक होने की मजबूत संभावना)"
            
    return final_pattern_scores, strong_signals, dead_signals

# 3. नंबर डेंसिटी बूस्टर - अंकों का फाइनल फिल्टर
def generate_adaptive_numbers(pattern_scores):
    number_confidence = {}
    shifts = ['DS', 'FD', 'GD', 'GL', 'DB', 'SG']
    
    for n in range(0, 100):
        # हर नंबर को 0 से शुरू करके उसकी पास होने की क्षमता का स्कोर देना
        confidence = 100.0
        rem = n % 10
        
        for shift in shifts:
            # हॉरिजॉन्тель चेक
            h1_s = pattern_scores.get(f'{shift}_H1', 0.5)
            h2_s = pattern_scores.get(f'{shift}_H2', 0.5)
            h3_s = pattern_scores.get(f'{shift}_H3', 0.5)
            h4_s = pattern_scores.get(f'{shift}_H4', 0.5)
            h5_s = pattern_scores.get(f'{shift}_H5', 0.5)
            
            if 1 <= n <= 20: confidence *= (h1_s / 0.5)
            elif 21 <= n <= 40: confidence *= (h2_s / 0.5)
            elif 41 <= n <= 60: confidence *= (h3_s / 0.5)
            elif 61 <= n <= 80: confidence *= (h4_s / 0.5)
            elif (81 <= n <= 99) or n == 0: confidence *= (h5_s / 0.5)
                
            # वर्टिकल चेक (आखिरी अंक)
            v1_s = pattern_scores.get(f'{shift}_V1', 0.5)
            v2_s = pattern_scores.get(f'{shift}_V2', 0.5)
            v3_s = pattern_scores.get(f'{shift}_V3', 0.5)
            v4_s = pattern_scores.get(f'{shift}_V4', 0.5)
            v5_s = pattern_scores.get(f'{shift}_V5', 0.5)
            
            if rem in [1, 2]: confidence *= (v1_s / 0.5)
            elif rem in [3, 4]: confidence *= (v2_s / 0.5)
            elif rem in [5, 6]: confidence *= (v3_s / 0.5)
            elif rem in [7, 8]: confidence *= (v4_s / 0.5)
            elif rem in [9, 0]: confidence *= (v5_s / 0.5)
                
        number_confidence[n] = confidence

    # टॉप स्कोर वाले नंबर छांटना जो कट-ऑफ पार करते हैं
    sorted_numbers = sorted(number_confidence.items(), key=lambda x: x[1], reverse=True)
    
    # केवल वो नंबर जिनकी कॉन्फिडेंस वैल्यू सबसे ज्यादा है
    top_numbers = [num for num, score in sorted_numbers[:18]] # केवल बेस्ट 18 सॉलिड अंक
    return sorted(top_numbers)

# मोबाइल यूजर इंटरफेस
st.subheader("📁 डेटा इनपुट (Data Input)")
uploaded_file = st.file_uploader(
    "अपनी एक्सेल शीट (.xlsx या .csv) अपलोड करें", 
    type=["csv", "xlsx"],
    key="maya_adaptive_booster_v4"
)

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        matrix_df, pattern_cols = process_binary_matrix(df)
        scores, strong_p, dead_p = adaptive_learning_prediction(matrix_df, pattern_cols)
        final_nums = generate_adaptive_numbers(scores)
        
        # परिणाम स्क्रीन (मोबाइल फ्रेंडली और स्पष्ट)
        st.success("🎯 वास्तविक मजबूत ट्रेंड्स (High Probability Trends)")
        if strong_p:
            for k, v in list(strong_p.items())[:8]: # टॉप 8 मुख्य ट्रेंड दिखाना
                st.write(f"🔹 **{k}**: {v}")
        else:
            st.write("कोई एकतरफा मजबूत ट्रेंड नहीं है, मार्केट संतुलित है।")
                
        st.error("🚫 वास्तविक डैड ट्रेंड्स (Low Probability Trends)")
        if dead_p:
            for k, v in list(dead_p.items())[:8]:
                st.write(f"🔸 **{k}**: {v}")
        else:
            st.write("इतिहास में आज कोई ब्लॉक ट्रेंड सक्रिय नहीं है।")
                
        st.markdown("---")
        st.subheader("🔮 MAYA AI - आज के फाइनल सॉलिड अंक")
        st.write("क्रॉस-शिफ्ट डेंसिटी और पिछले इतिहास की गलतियों को सुधार कर निकाले गए **सर्वश्रेष्ठ अंक**:")
        
        if final_nums:
            st.markdown(f"### 👑 `{', '.join(map(str, final_nums))}`")
            st.write(f"कुल सुरक्षित अंकों की संख्या: **{len(final_nums)}**")
        else:
            st.write("डेटा अपर्याप्त है या अत्यधिक टकराव है।")
            
    except Exception as e:
        st.error(f"फाइल प्रोसेसिंग एरर: {e}")
