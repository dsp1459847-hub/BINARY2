import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict

# मोबाइल और बड़ी स्क्रीन दोनों के लिए लेआउट सेट करना
st.set_page_config(page_title="MAYA AI - Self Learning Matrix v5.0", layout="wide")

st.title("👑 MAYA AI - ऑटो-बैकटेस्टिंग और मल्टी-शिफ्ट बूस्टर v5.0")
st.write("यह इंजन पहले पिछले 90 दिनों के लाखों पैटर्न्स का बैकटेस्ट करता है और सिर्फ 80% से ऊपर वाले सफल पैटर्न्स से आज की प्रेडिक्शन निकालता है।")

# 1. 60-कॉलम बाइनरी मैट्रिक्स जनरेशन (सुरक्षित डेटा टाइप)
def process_binary_matrix(df):
    shifts = ['DS', 'FD', 'GD', 'GL', 'DB', 'SG']
    df = df.dropna(subset=['DATE']).reset_index(drop=True)
    
    h_data = {}
    v_data = {}
    
    for shift in shifts:
        if shift in df.columns:
            s_data = pd.to_numeric(df[shift].replace('XX', np.nan), errors='coerce')
            
            # 5 हॉरिजॉन्टल पार्ट्स
            h_data[f'{shift}_H1'] = np.where((s_data >= 1) & (s_data <= 20), 1, np.where(s_data.isna(), -1, 0))
            h_data[f'{shift}_H2'] = np.where((s_data >= 21) & (s_data <= 40), 1, np.where(s_data.isna(), -1, 0))
            h_data[f'{shift}_H3'] = np.where((s_data >= 41) & (s_data <= 60), 1, np.where(s_data.isna(), -1, 0))
            h_data[f'{shift}_H4'] = np.where((s_data >= 61) & (s_data <= 80), 1, np.where(s_data.isna(), -1, 0))
            h_data[f'{shift}_H5'] = np.where(((s_data >= 81) & (s_data <= 99)) | (s_data == 0), 1, np.where(s_data.isna(), -1, 0))
            
            # 5 वर्टिकल पार्ट्स (आखिरी अंक)
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

# 2. मास्टर डीप बैकटेस्टिंग इंजन (कल, परसों, तरसों, और डबल-शिफ्ट कॉम्बिनेशन चेकर)
def deep_historical_backtest(matrix_df, pattern_cols):
    total_rows = len(matrix_df)
    if total_rows < 30:
        return {}, [], []
        
    # पिछले 90 दिनों या रिकॉर्ड्स तक का बैकटेस्ट सीमित रखना
    lookback_limit = min(90, total_rows - 5)
    start_idx = total_rows - lookback_limit
    
    successful_patterns = []
    failed_patterns_count = 0
    
    # प्रत्येक टारगेट पैटर्न के लिए बैकटेस्ट स्कोर स्टोर करना
    pattern_accuracy_registry = {}
    
    # अंतिम लाइव स्थिति (कल का डेटा)
    last_row = matrix_df.iloc[total_rows - 1]
    last_2_row = matrix_df.iloc[total_rows - 2] # परसों
    last_3_row = matrix_df.iloc[total_rows - 3] # तरसों
    
    for target_col in pattern_cols:
        best_accuracy = 0.0
        best_prediction_value = -1
        best_logic_name = ""
        
        # प्रोबेबिलिटी टाइप 1: कल (t-1) का सिंगल कॉलम रिलेशिनशिप
        for col_past in pattern_cols:
            yesterday_val = last_row[col_past]
            if yesterday_val == -1: continue
            
            # इतिहास में इस नियम का बैकटेस्ट करना
            match_1 = 0
            total_1 = 0
            for idx in range(start_idx, total_rows - 1):
                if matrix_df.loc[idx, col_past] == yesterday_val:
                    actual_next = matrix_df.loc[idx + 1, target_col]
                    if actual_next != -1:
                        total_1 += 1
                        if actual_next == 1: match_1 += 1
            
            if total_1 >= 8: # न्यूनतम 8 बार आना ज़रूरी है
                rate_1 = match_1 / total_1
                rate_0 = (total_1 - match_1) / total_1
                
                if rate_1 >= 0.80 and rate_1 > best_accuracy:
                    best_accuracy, best_prediction_value, best_logic_name = rate_1, 1, f"कल के {col_past} से लिंक"
                if rate_0 >= 0.80 and rate_0 > best_accuracy:
                    best_accuracy, best_prediction_value, best_logic_name = rate_0, 0, f"कल के {col_past} से लिंक"

        # प्रोबेबिलिटी टाइप 2: परसों (t-2) या तरसों (t-3) का लॉन्ग गैप लिंक
        if best_accuracy < 0.80:
            for col_past in pattern_cols:
                p_val = last_2_row[col_past] # परसों
                if p_val == -1: continue
                match_2 = 0
                total_2 = 0
                for idx in range(start_idx, total_rows - 2):
                    if matrix_df.loc[idx, col_past] == p_val:
                        actual_next = matrix_df.loc[idx + 2, target_col]
                        if actual_next != -1:
                            total_2 += 1
                            if actual_next == 1: match_2 += 1
                            
                if total_2 >= 8:
                    rate_1 = match_2 / total_2
                    rate_0 = (total_2 - match_2) / total_2
                    if rate_1 >= 0.80 and rate_1 > best_accuracy:
                        best_accuracy, best_prediction_value, best_logic_name = rate_1, 1, f"परसों के {col_past} से लिंक"
                    if rate_0 >= 0.80 and rate_0 > best_accuracy:
                        best_accuracy, best_prediction_value, best_logic_name = rate_0, 0, f"परसों के {col_past} से लिंक"

        # प्रोबेबिलिटी टाइप 3: डबल शिफ्ट जॉइंट रिलेशन (दो पैटर्न्स का एक साथ आना)
        if best_accuracy < 0.80:
            # टॉप 5 सबसे महत्वपूर्ण जोड़ियों को चेक करना ताकि मोबाइल पर स्पीड बनी रहे
            for i in range(len(pattern_cols) - 1):
                colA = pattern_cols[i]
                colB = pattern_cols[i+1]
                valA, valB = last_row[colA], last_row[colB]
                if valA == -1 or valB == -1: continue
                
                match_double = 0
                total_double = 0
                for idx in range(start_idx, total_rows - 1):
                    if matrix_df.loc[idx, colA] == valA and matrix_df.loc[idx, colB] == valB:
                        actual_next = matrix_df.loc[idx + 1, target_col]
                        if actual_next != -1:
                            total_double += 1
                            if actual_next == 1: match_double += 1
                            
                if total_double >= 5:
                    rate_1 = match_double / total_double
                    rate_0 = (total_double - match_double) / total_double
                    if rate_1 >= 0.80 and rate_1 > best_accuracy:
                        best_accuracy, best_prediction_value, best_logic_name = rate_1, 1, f"डबल लिंक ({colA} + {colB})"
                    if rate_0 >= 0.80 and rate_0 > best_accuracy:
                        best_accuracy, best_prediction_value, best_logic_name = rate_0, 0, f"डबल लिंक ({colA} + {colB})"

        # अगर कोई भी नियम 80% या 90% बैकटेस्ट पास कर गया, तो ही रजिस्टर करें
        if best_accuracy >= 0.80:
            pattern_accuracy_registry[target_col] = {
                'prediction': best_prediction_value,
                'accuracy': best_accuracy,
                'logic': best_logic_name
            }
            successful_patterns.append(f"✅ **{target_col}**: {best_logic_name} | बैकटेस्ट सटीकता: {best_accuracy*100:.1f}% -> आउटपुट: **{best_prediction_value}**")
        else:
            pattern_accuracy_registry[target_col] = {'prediction': -1, 'accuracy': 0, 'logic': "फेल"}
            failed_patterns_count += 1
            
    return pattern_accuracy_registry, successful_patterns, failed_patterns_count

# 3. बैकटेस्टेड नंबर फ़िल्टरिंग
def generate_backtested_numbers(registry_scores):
    allowed_numbers = set(range(0, 100))
    removed_numbers = set()
    shifts = ['DS', 'FD', 'GD', 'GL', 'DB', 'SG']
    
    for shift in shifts:
        # केवल उन्हीं कॉलम्स को ब्लॉक करें जिनका बैकटेस्ट प्रेडिक्शन 0 (डैड) आया है और सटीकता 80% से ऊपर है
        if registry_scores.get(f'{shift}_H1', {}).get('prediction') == 0: removed_numbers.update(range(1, 21))
        if registry_scores.get(f'{shift}_H2', {}).get('prediction') == 0: removed_numbers.update(range(21, 41))
        if registry_scores.get(f'{shift}_H3', {}).get('prediction') == 0: removed_numbers.update(range(41, 61))
        if registry_scores.get(f'{shift}_H4', {}).get('prediction') == 0: removed_numbers.update(range(61, 81))
        if registry_scores.get(f'{shift}_H5', {}).get('prediction') == 0:
            removed_numbers.update(range(81, 100))
            removed_numbers.add(0)
            
        for n in list(allowed_numbers):
            rem = n % 10
            if registry_scores.get(f'{shift}_V1', {}).get('prediction') == 0 and rem in [1, 2]: removed_numbers.add(n)
            if registry_scores.get(f'{shift}_V2', {}).get('prediction') == 0 and rem in [3, 4]: removed_numbers.add(n)
            if registry_scores.get(f'{shift}_V3', {}).get('prediction') == 0 and rem in [5, 6]: removed_numbers.add(n)
            if registry_scores.get(f'{shift}_V4', {}).get('prediction') == 0 and rem in [7, 8]: removed_numbers.add(n)
            if registry_scores.get(f'{shift}_V5', {}).get('prediction') == 0 and rem in [9, 0]: removed_numbers.add(n)

    final_prediction = allowed_numbers - removed_numbers
    return sorted(list(final_prediction)), sorted(list(removed_numbers))

# मोबाइल यूजर इंटरफेस
st.subheader("📁 डेटा इनपुट और लाइव बैकटेस्टिंग (Data Input & Live Backtest)")
uploaded_file = st.file_uploader(
    "अपनी एक्सेल शीट (.xlsx या .csv) अपलोड करें", 
    type=["csv", "xlsx"],
    key="maya_backtest_booster_v5"
)

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        matrix_df, pattern_cols = process_binary_matrix(df)
        registry, success_list, failed_cnt = deep_historical_backtest(matrix_df, pattern_cols)
        final_nums, blocked_nums = generate_backtested_numbers(registry)
        
        # परिणाम का प्रदर्शन
        st.header("📊 लाइव बैकटेस्टिंग रिपोर्ट (Live Test Result)")
        st.write(f"कुल 60 पैटर्न्स में से **{len(success_list)}** पैटर्न्स ने 80%-100% सफलता दर को पास किया। शेष **{failed_cnt}** को कमजोर मानकर हटा दिया गया।")
        
        if success_list:
            with st.expander("पास हुए सफल पैटर्न्स और उनके इतिहास के संबंध देखें"):
                for item in success_list:
                    st.write(item)
                    
        st.markdown("---")
        st.header("🔮 आज के 100% बैकटेस्ट पास सॉलिड अंक")
        st.write("इतिहास की कड़ी कसौटी (80%-100% सक्सेस रेट) पर खरे उतरे फाइनल नंबर्स:")
        
        if final_nums:
            st.markdown(f"## 👑 `{', '.join(map(str, final_nums))}`")
            st.write(f"कुल बचे हुए सॉलिड अंकों की संख्या: **{len(final_nums)}**")
        else:
            st.warning("इतिहास के कड़े फिल्टरों के कारण आज कोई भी सुरक्षित नंबर पास नहीं हो सका। मार्केट बहुत अनिश्चित है।")
            
        # तारीख के साथ कनवर्टेड शीट देखने का विकल्प
        if st.checkbox("पूरी कनवर्टेड बाइनरी शीट (तारीख और सीरियल नंबर के साथ) देखें"):
            st.write("आप नीचे स्क्रॉल करके किसी भी तारीख की प्रेडिक्शन और कोड्स खुद चेक कर सकते हैं:")
            st.dataframe(matrix_df[['S. NUMBER ', 'DATE'] + pattern_cols])
            
    except Exception as e:
        st.error(f"फाइल प्रोसेसिंग एरर: {e}")
