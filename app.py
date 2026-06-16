import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter

# मोबाइल स्क्रीन के अनुकूल लेआउट सेट करना
st.set_page_config(page_title="MAYA AI - Matrix v2.0", layout="centered")

st.title("📊 MAYA AI - मोबाइल बाइनरी बूस्टर v2.0")
st.write("मोबाइल से एक्सेल शीट अपलोड करें और आज की प्रेडिक्शन निकालें।")

# 1. डेटा को प्रोसेस करके 60 बाइनरी कॉलम्स (Matrix) बनाने का फंक्शन (एरर-फ्री वर्जन)
def process_binary_matrix(df):
    shifts = ['DS', 'FD', 'GD', 'GL', 'DB', 'SG']
    df = df.dropna(subset=['DATE']).reset_index(drop=True)
    
    h_data = {}
    v_data = {}
    
    for shift in shifts:
        if shift in df.columns:
            # XX या खाली सेल को हटाकर केवल नंबर्स को प्रोसेस करना
            s_data = pd.to_numeric(df[shift].replace('XX', np.nan), errors='coerce')
            
            # .astype(object) का उपयोग किया ताकि XX, 1 और 0 एक साथ रहने पर NumPy क्रैश न हो
            # 5 हॉरिजॉन्टल पार्ट्स (1-20, 21-40, 41-60, 61-80, 81-99/0)
            h_data[f'{shift}_H1'] = np.where((s_data >= 1) & (s_data <= 20), "1", np.where(s_data.isna(), "XX", "0")).astype(object)
            h_data[f'{shift}_H2'] = np.where((s_data >= 21) & (s_data <= 40), "1", np.where(s_data.isna(), "XX", "0")).astype(object)
            h_data[f'{shift}_H3'] = np.where((s_data >= 41) & (s_data <= 60), "1", np.where(s_data.isna(), "XX", "0")).astype(object)
            h_data[f'{shift}_H4'] = np.where((s_data >= 61) & (s_data <= 80), "1", np.where(s_data.isna(), "XX", "0")).astype(object)
            h_data[f'{shift}_H5'] = np.where(((s_data >= 81) & (s_data <= 99)) | (s_data == 0), "1", np.where(s_data.isna(), "XX", "0")).astype(object)
            
            # 5 वर्टिकल पार्ट्स (आखिरी अंक 1-2, 3-4, 5-6, 7-8, 9-0)
            last_digit = s_data % 10
            v_data[f'{shift}_V1'] = np.where((last_digit == 1) | (last_digit == 2), "1", np.where(s_data.isna(), "XX", "0")).astype(object)
            v_data[f'{shift}_V2'] = np.where((last_digit == 3) | (last_digit == 4), "1", np.where(s_data.isna(), "XX", "0")).astype(object)
            v_data[f'{shift}_V3'] = np.where((last_digit == 5) | (last_digit == 6), "1", np.where(s_data.isna(), "XX", "0")).astype(object)
            v_data[f'{shift}_V4'] = np.where((last_digit == 7) | (last_digit == 8), "1", np.where(s_data.isna(), "XX", "0")).astype(object)
            v_data[f'{shift}_V5'] = np.where((last_digit == 9) | (last_digit == 0), "1", np.where(s_data.isna(), "XX", "0")).astype(object)
            
    h_df = pd.DataFrame(h_data)
    v_df = pd.DataFrame(v_data)
    
    full_matrix = pd.concat([df[['S. NUMBER ', 'DATE'] + [s for s in shifts if s in df.columns]], h_df, v_df], axis=1)
    return full_matrix, list(h_df.columns) + list(v_df.columns)

# 2. प्रेडिक्शन इंजन: कल, परसों, तरसों के 3-दिन के सीक्वेंस को ट्रेस करना
def analyze_and_predict(matrix_df, pattern_cols):
    predictions = {}
    strong_patterns = {}
    dead_patterns = {}
    
    for col in pattern_cols:
        # केवल वैलिड 0 और 1 डेटा को उठाना, XX को छोड़ देना (यहाँ स्ट्रिंग और नंबर दोनों को सेफली चेक किया है)
        valid_rows = matrix_df[matrix_df[col].isin([0, 1, '0', '1', 0.0, 1.0])]
        col_data = valid_rows[col].astype(int).tolist()
        
        if len(col_data) < 5:
            predictions[col] = "XX"
            continue
            
        # कल, परसों, तरसों का सीक्वेंस और उसका अगला टारगेट बनाना
        sequences = []
        targets = []
        for i in range(len(col_data) - 3):
            seq = (col_data[i], col_data[i+1], col_data[i+2])
            target = col_data[i+3]
            sequences.append(seq)
            targets.append(target)
            
        # बिल्कुल हालिया 3 दिनों का सीक्वेंस क्या बना है
        last_3_days_seq = (col_data[-3], col_data[-2], col_data[-1])
        
        # इतिहास में इस सीक्वेंस के बाद क्या आया, उसे मैच करना
        match_targets = [targets[i] for i in range(len(sequences)) if sequences[i] == last_3_days_seq]
        
        if match_targets:
            counts = Counter(match_targets)
            prob_1 = counts[1] / len(match_targets)
            prob_0 = counts[0] / len(match_targets)
            
            # 70% या उससे ऊपर की संभावना को सॉलिड मानना
            if prob_1 >= 0.70:
                predictions[col] = 1
                strong_patterns[col] = f"क्रम {last_3_days_seq} के बाद '1' आने के चांस {prob_1*100:.0f}% हैं।"
            elif prob_0 >= 0.70:
                predictions[col] = 0
                dead_patterns[col] = f"क्रम {last_3_days_seq} के बाद '0' आने के चांस {prob_0*100:.0f}% हैं।"
            else:
                predictions[col] = "कन्फ्यूज्ड"
        else:
            # नया कॉम्बिनेशन होने पर ओवरऑल इतिहास का एवरेज लेना
            predictions[col] = 1 if np.mean(col_data) > 0.5 else 0
            
    return predictions, strong_patterns, dead_patterns

# 3. बाइनरी प्रेडिक्शन को वापस 0-99 के अंकों में फ़िल्टर करने का लॉजिक
def generate_final_numbers(predictions):
    allowed_numbers = set(range(0, 100))
    removed_numbers = set()
    shifts = ['DS', 'FD', 'GD', 'GL', 'DB', 'SG']
    
    for shift in shifts:
        # हॉरिजॉन्टल ब्लॉकिंग (अगर किसी पार्ट का प्रेडिक्शन 0 है तो उसके सारे अंक हटा दो)
        if predictions.get(f'{shift}_H1') == 0: removed_numbers.update(range(1, 21))
        if predictions.get(f'{shift}_H2') == 0: removed_numbers.update(range(21, 41))
        if predictions.get(f'{shift}_H3') == 0: removed_numbers.update(range(41, 61))
        if predictions.get(f'{shift}_H4') == 0: removed_numbers.update(range(61, 81))
        if predictions.get(f'{shift}_H5') == 0:
            removed_numbers.update(range(81, 100))
            removed_numbers.add(0)
            
        # वर्टिकल ब्लॉकिंग (आखिरी अंक के आधार पर अंकों को हटाना)
        for n in list(allowed_numbers):
            rem = n % 10
            if predictions.get(f'{shift}_V1') == 0 and rem in [1, 2]: removed_numbers.add(n)
            if predictions.get(f'{shift}_V2') == 0 and rem in [3, 4]: removed_numbers.add(n)
            if predictions.get(f'{shift}_V3') == 0 and rem in [5, 6]: removed_numbers.add(n)
            if predictions.get(f'{shift}_V4') == 0 and rem in [7, 8]: removed_numbers.add(n)
            if predictions.get(f'{shift}_V5') == 0 and rem in [9, 0]: removed_numbers.add(n)

    final_prediction = allowed_numbers - removed_numbers
    return sorted(list(final_prediction)), sorted(list(removed_numbers))

# मोबाइल अपलोडर इंटरफेस
st.subheader("📁 डेटा इनपुट (Data Input)")
uploaded_file = st.file_uploader(
    "अपनी एक्सेल शीट (.xlsx या .csv) अपलोड करें", 
    type=["csv", "xlsx"],
    key="maya_file_uploader_v2"
)

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        matrix_df, pattern_cols = process_binary_matrix(df)
        predictions, strong_p, dead_p = analyze_and_predict(matrix_df, pattern_cols)
        final_nums, blocked_nums = generate_final_numbers(predictions)
        
        # मोबाइल व्यू के लिए एक के नीचे एक साफ-साफ डिस्प्ले करना
        st.success("🎯 मजबूत पैटर्न्स (Strong Shifts)")
        if strong_p:
            for k, v in strong_p.items():
                st.write(f"🔹 **{k}**: {v}")
        else:
            st.write("कोई खास मजबूत पैटर्न नहीं मिला।")
                
        st.error("🚫 डैड पैटर्न्स (हटाने योग्य Shifts)")
        if dead_p:
            for k, v in dead_p.items():
                st.write(f"🔸 **{k}**: {v}")
        else:
            st.write("कोई डैड पैटर्न एक्टिव नहीं है।")
                
        st.markdown("---")
        st.subheader("🔮 आज के फाइनल सॉलिड अंक (Filtered Numbers)")
        st.write("कमजोर और डैड पैटर्न्स को हटाकर बचे हुए सॉलिड नंबरों की लिस्ट:")
        
        if final_nums:
            # बड़े अक्षरों में दिखाना ताकि मोबाइल पर साफ दिखे
            st.markdown(f"### 👑 `{', '.join(map(str, final_nums))}`")
            st.write(f"कुल बचे हुए अंकों की संख्या: **{len(final_nums)}**")
        else:
            st.write("पैटर्न्स के अत्यधिक टकराव के कारण कोई नंबर नहीं बचा।")
            
        if st.checkbox("पूरी कनवर्टेड बाइनरी शीट (60 कॉलम) देखें"):
            st.dataframe(matrix_df)
            
    except Exception as e:
        st.error(f"फाइल प्रोसेसिंग एरर: {e}")
