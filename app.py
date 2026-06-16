import streamlit as st
import pandas as pd
import numpy as np

# मोबाइल और डेस्कटॉप दोनों के लिए बेस्ट लेआउट
st.set_page_config(page_title="MAYA AI - Matrix Multi-Gap v5.5", layout="wide")

st.title("👑 MAYA AI - टाइम-गैप क्रॉस-शिफ्ट बूस्टर v5.5")
st.write("एक शिफ्ट से दूसरी शिफ्ट में दिनों के गैप (1, 2, 3, 4 दिन) का विश्लेषण करके 100% बैकटेस्टेड अंक निकालना।")

# 1. 60-कॉलम बाइनरी मैट्रिक्स जनरेशन (डेटा क्लीनिंग के साथ)
def process_binary_matrix(df):
    shifts = ['DS', 'FD', 'GD', 'GL', 'DB', 'SG']
    df = df.dropna(subset=['DATE']).reset_index(drop=True)
    
    h_data = {}
    v_data = {}
    
    for shift in shifts:
        if shift in df.columns:
            s_data = pd.to_numeric(df[shift].replace('XX', np.nan), errors='coerce')
            
            # हॉरिजॉन्टल पार्ट्स (1-20, 21-40, 41-60, 61-80, 81-99/0)
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

# 2. क्रॉस-डेट शिफ्ट और मल्टी-गैप मैचिंग इंजन
def analyze_multi_gap_relations(matrix_df, pattern_cols):
    total_rows = len(matrix_df)
    if total_rows < 15:
        return {}, [], []
        
    predictions = {}
    winner_log_list = []
    dead_log_list = []
    
    # टारगेट हमेशा बेस शिफ्ट (DS) के पैटर्न्स होंगे
    base_target_cols = [c for c in pattern_cols if c.startswith('DS_')]
    
    # अन्य सभी शिफ्ट्स जिनसे लिंक ढूंढना है (FD, GD, GL, DB, SG)
    other_source_cols = [c for c in pattern_cols if not c.startswith('DS_')]
    
    # पिछले 90 दिनों का बैकटेस्ट दायरा
    lookback_limit = min(90, total_rows - 6)
    start_idx = total_rows - lookback_limit
    
    for target_col in base_target_cols:
        found_perfect_logic = False
        
        # हम 1 दिन से लेकर 4 दिन तक के गैप (Time Gaps) को चेक करेंगे
        for gap in [1, 2, 3, 4]:
            if found_perfect_logic: break
            
            for src_col in other_source_cols:
                # कल/परसों/तरसों/नरसों के गैप के आधार पर पिछली वैल्यू उठाना
                past_live_val = matrix_df.loc[total_rows - gap, src_col]
                if past_live_val == -1: continue # छुट्टी है तो छोड़ें
                
                match_count_1 = 0
                total_match_count = 0
                
                # इतिहास में इस विशिष्ट टाइम-गैप के नियम का टेस्ट करना
                for idx in range(start_idx, total_rows - gap):
                    if matrix_df.loc[idx, src_col] == past_live_val:
                        actual_future_val = matrix_df.loc[idx + gap, target_col]
                        if actual_future_val != -1:
                            total_match_count += 1
                            if actual_future_val == 1:
                                match_count_1 += 1
                                
                # कड़ा फ़िल्टर: न्यूनतम 5 बार इतिहास में आना ज़रूरी है और परिणाम 100% होना चाहिए
                if total_match_count >= 5:
                    success_rate_1 = match_count_1 / total_match_count
                    
                    # 100% आने का पैटर्न (Winner)
                    if success_rate_1 == 1.0:
                        predictions[target_col] = 1
                        winner_log_list.append(f"🎯 **{target_col}** -> {gap} दिन पहले के **{src_col}** से लिंक | इतिहास में {total_match_count} बार में से {match_count_1} बार '1' आया (100% पास)")
                        found_perfect_logic = True
                        break
                        
                    # 100% ना आने का पैटर्न (Dead/Zero)
                    elif success_rate_1 == 0.0:
                        predictions[target_col] = 0
                        dead_log_list.append(f"🚫 **{target_col}** -> {gap} दिन पहले के **{src_col}** से लिंक | इतिहास में {total_match_count} बार में से 0 बार '1' आया (100% ब्लॉक)")
                        found_perfect_logic = True
                        break
                        
        # अगर किसी पैटर्न का कोई 100% इतिहास नहीं मिला, तो उसे न्यूट्रल (-1) छोड़ दें (कोई फर्जी दावा नहीं)
        if not found_perfect_logic:
            predictions[target_col] = -1
            
    return predictions, winner_log_list, dead_log_list

# 3. सॉलिड अंकों की फ़िल्टरिंग थ्योरी
def filter_numbers_by_gaps(predictions):
    allowed_numbers = set(range(0, 100))
    removed_numbers = set()
    
    # 100% ब्लॉक (0 वाले) पैटर्न्स के आधार पर अंकों को हटाना
    if predictions.get('DS_H1') == 0: removed_numbers.update(range(1, 21))
    if predictions.get('DS_H2') == 0: removed_numbers.update(range(21, 41))
    if predictions.get('DS_H3') == 0: removed_numbers.update(range(41, 61))
    if predictions.get('DS_H4') == 0: removed_numbers.update(range(61, 81))
    if predictions.get('DS_H5') == 0:
        removed_numbers.update(range(81, 100))
        removed_numbers.add(0)
        
    for n in list(allowed_numbers):
        rem = n % 10
        if predictions.get('DS_V1') == 0 and rem in [1, 2]: removed_numbers.add(n)
        if predictions.get('DS_V2') == 0 and rem in [3, 4]: removed_numbers.add(n)
        if predictions.get('DS_V3') == 0 and rem in [5, 6]: removed_numbers.add(n)
        if predictions.get('DS_V4') == 0 and rem in [7, 8]: removed_numbers.add(n)
        if predictions.get('DS_V5') == 0 and rem in [9, 0]: removed_numbers.add(n)

    final_prediction = allowed_numbers - removed_numbers
    return sorted(list(final_prediction)), sorted(list(removed_numbers))

# मोबाइल अपलोडर इंटरफेस
st.subheader("📁 डेटा इनपुट (Data Input)")
uploaded_file = st.file_uploader(
    "अपनी एक्सेल शीट (.xlsx या .csv) अपलोड करें", 
    type=["csv", "xlsx"],
    key="maya_gap_booster_v5.5"
)

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        matrix_df, pattern_cols = process_binary_matrix(df)
        predictions, winners, deads = analyze_multi_gap_relations(matrix_df, pattern_cols)
        final_nums, blocked_nums = filter_numbers_by_gaps(predictions)
        
        # मोबाइल के अनुकूल साफ डिस्प्ले
        st.header("📊 टाइम-गैप मैचिंग विश्लेषण")
        
        col1, col2 = st.columns(2)
        with col1:
            st.success("🎯 100% मैचिंग पैटर्न्स (Winner - Output 1)")
            if winners:
                for w in winners: st.write(w)
            else:
                st.write("आज इतिहास में 100% मैच होने वाला कोई विनर पैटर्न नहीं मिला।")
                
        with col2:
            st.error("🚫 100% ब्लॉक पैटर्न्स (Dead - Output 0)")
            if deads:
                for d in deads: st.write(d)
            else:
                st.write("आज इतिहास में 100% ब्लॉक होने वाला कोई डैड पैटर्न नहीं मिला।")
                
        st.markdown("---")
        st.header("🔮 आज के फाइनल सॉलिड अंक (Base Shift: DS)")
        st.write("आपके बताए गए क्रॉस्ड टाइम-गैप नियमों के आधार पर छांटे गए नंबर्स:")
        
        if len(blocked_nums) > 0:
            st.markdown(f"### 👑 `{', '.join(map(str, final_nums))}`")
            st.write(f"कुल सुरक्षित अंकों की संख्या: **{len(final_nums)}** (बाकी {len(blocked_nums)} अंक 100% ब्लॉक पैटर्न्स के कारण हटा दिए गए)")
        else:
            st.warning("कोई भी पैटर्न 100% ब्लॉक साबित नहीं हुआ, इसलिए सभी 100 अंक सुरक्षित क्षेत्र में हैं। कृपया अधिक डेटा जोड़ें।")
            
        if st.checkbox("तारीख के साथ पूरी 60-कॉलम कनवर्टेड शीट देखें"):
            st.dataframe(matrix_df[['S. NUMBER ', 'DATE'] + pattern_cols])
            
    except Exception as e:
        st.error(f"फाइल प्रोसेसिंग एरर: {e}")
