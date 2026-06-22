import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# 폰트 설정 (웹앱 그래프 한글 깨짐 방지)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# layout='wide' 로 변경하여 우측 결과를 더 시원하게 볼 수 있도록 세팅
st.set_page_config(page_title="항공편 지연 예측 대시보드", layout="wide")

model_path = "flight_pipeline.pkl"

st.title("항공편 지연 확률 예측 대시보드")
st.markdown("""
이 대시보드는 비행 데이터를 학습한 기계학습 모델의 예측 결과를 제공합니다.
좌측 사이드바에서 비행 조건을 설정하시면, 메인 화면에서 AI의 예측 결과와 분석 근거(시각화 차트)를 확인할 수 있습니다.
""")
st.divider()

if not os.path.exists(model_path):
    st.error("모델을 찾을 수 없습니다.\n\nVS Code에서 `항공편_지연_예측.ipynb` 파일을 열어 모두 실행(Run All)을 누르시면 모델 파일이 생성됩니다. 생성된 후 앱을 새로고침 해주세요.")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(model_path)

pipeline = load_model()

# 주요 공항 및 항공사 목록
airports = [
    'ATL', 'DFW', 'DEN', 'ORD', 'LAX', 'JFK', 'LAS', 'MCO', 'MIA', 'SEA', 
    'PHX', 'EWR', 'SFO', 'IAH', 'BOS', 'SLC', 'BWI', 'IAD', 'SAN', 'HNL'
]
airlines = [
    'Delta Air Lines Inc.', 'American Airlines Inc.', 'United Air Lines Inc.', 
    'Southwest Airlines Co.', 'JetBlue Airways', 'SkyWest Airlines Inc.',
    'Alaska Airlines Inc.', 'Spirit Air Lines'
]

# 사이드바로 모든 입력 폼 이동
with st.sidebar:
    st.header("비행 조건 설정")
    
    st.subheader("공항 및 항공사")
    origin = st.selectbox("출발 공항 (Origin)", airports, index=0)
    destination = st.selectbox("도착 공항 (Destination)", airports, index=4)
    airline = st.selectbox("항공사 (Airline)", airlines, index=0)
    
    st.divider()
    
    st.subheader("일정 및 거리")
    month = st.slider("월 (Month)", 1, 12, 6)
    hour = st.slider("출발 예정 시간 (Hour)", 0, 23, 14)
    distance = st.number_input("비행 거리 (miles)", min_value=100, max_value=5000, value=800, step=50)
    
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("지연 확률 예측하기", type="primary", use_container_width=True)

# 메인 화면 처리 (우측)
if origin == destination:
    st.warning("출발 공항과 도착 공항이 동일합니다. 좌측 사이드바에서 다시 선택해 주세요.")
else:
    if predict_btn:
        input_data = pd.DataFrame([{
            'Month': month,
            'Distance': distance,
            'Airline': airline,
            'Origin_Airport': origin,
            'Destination_Airport': destination,
            'Hour': hour
        }])
        
        with st.spinner("AI 모델이 예측을 수행 중입니다..."):
            delay_prob = pipeline.predict_proba(input_data)[0][1]
        
        st.subheader("인공지능 예측 결과")
        
        st.metric(label="지연 및 결항 확률", value=f"{delay_prob * 100:.1f}%")
        
        if delay_prob > 0.6:
            st.error("위험: 해당 조건에서 비행기가 지연될 확률이 매우 높습니다. (XGBoost 위험 패턴 감지)")
        elif delay_prob > 0.4:
            st.warning("주의: 약간의 지연 가능성이 존재합니다.")
        else:
            st.success("정상 운항 예상: 정시 출발 확률이 높습니다.")
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("왜 이런 예측이 나왔을까요? (AI 분석 근거)")
        
        try:
            model = pipeline.named_steps['model']
            preprocessor = pipeline.named_steps['preprocessor']
            
            num_features = ['Month', 'Distance', 'Hour']
            cat_features = ['Airline', 'Origin_Airport', 'Destination_Airport']
            
            cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
            cat_feature_names = cat_encoder.get_feature_names_out(cat_features)
            feature_names = num_features + list(cat_feature_names)
            
            importances = model.feature_importances_
            
            imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            imp_df['Feature'] = imp_df['Feature'].apply(lambda x: x.split('_', 1)[-1] if '_' in x and not x.startswith('Origin') and not x.startswith('Dest') else x)
            imp_df = imp_df.sort_values(by='Importance', ascending=False).head(5)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x='Importance', y='Feature', data=imp_df, palette='Reds_r' if delay_prob > 0.5 else 'Blues_r', ax=ax)
            ax.set_title("해당 예측에 가장 큰 영향을 미친 5대 핵심 요인", pad=15)
            ax.set_xlabel("영향도 (Feature Importance)")
            ax.set_ylabel("")
            
            st.pyplot(fig)
            st.caption("XGBoost 알고리즘이 입력된 데이터를 분석하여 도출한 결정 요인들입니다.")
        except Exception as e:
            st.info("시각화 데이터를 불러오는 중 오류가 발생했습니다.")
    else:
        st.info("좌측 사이드바에서 비행 조건을 설정한 후 '지연 확률 예측하기' 버튼을 눌러주세요.")
