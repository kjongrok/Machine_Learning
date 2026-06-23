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

# 주요 공항 및 항공사 한글 매핑 딕셔너리
airport_names = {
    'ATL': '애틀랜타 (ATL)', 'DFW': '댈러스/포트워스 (DFW)', 'DEN': '덴버 (DEN)', 'ORD': '시카고 오헤어 (ORD)', 
    'LAX': '로스앤젤레스 (LAX)', 'JFK': '뉴욕 JFK (JFK)', 'LAS': '라스베이거스 (LAS)', 'MCO': '올랜도 (MCO)', 
    'MIA': '마이애미 (MIA)', 'SEA': '시애틀 (SEA)', 'PHX': '피닉스 (PHX)', 'EWR': '뉴어크 (EWR)', 
    'SFO': '샌프란시스코 (SFO)', 'IAH': '휴스턴 (IAH)', 'BOS': '보스턴 (BOS)', 'SLC': '솔트레이크시티 (SLC)', 
    'BWI': '볼티모어 (BWI)', 'IAD': '워싱턴 덜레스 (IAD)', 'SAN': '샌디에이고 (SAN)', 'HNL': '호놀룰루 (HNL)'
}

airline_names = {
    'Delta Air Lines Inc.': '델타 항공 (Delta)', 'American Airlines Inc.': '아메리칸 항공 (American)', 
    'United Air Lines Inc.': '유나이티드 항공 (United)', 'Southwest Airlines Co.': '사우스웨스트 항공 (Southwest)', 
    'JetBlue Airways': '제트블루 항공 (JetBlue)', 'SkyWest Airlines Inc.': '스카이웨스트 항공 (SkyWest)',
    'Alaska Airlines Inc.': '알래스카 항공 (Alaska)', 'Spirit Air Lines': '스피릿 항공 (Spirit)'
}

airports = list(airport_names.keys())
airlines = list(airline_names.keys())

# 사이드바로 모든 입력 폼 이동
with st.sidebar:
    st.header("비행 조건 설정")
    
    st.subheader("공항 및 항공사")
    origin = st.selectbox("출발 공항 (Origin)", airports, format_func=lambda x: airport_names[x], index=0)
    destination = st.selectbox("도착 공항 (Destination)", airports, format_func=lambda x: airport_names[x], index=4)
    airline = st.selectbox("항공사 (Airline)", airlines, format_func=lambda x: airline_names[x], index=0)
    
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
        
        tab1, tab2, tab3 = st.tabs(["📊 실시간 예측 결과", "🧠 AI 원인 분석", "📈 데이터 및 모델 정보"])
        
        with tab1:
            st.subheader("인공지능 예측 결과")
            st.metric(label="지연 및 결항 확률", value=f"{delay_prob * 100:.1f}%")
            
            if delay_prob > 0.6:
                st.error("위험: 해당 조건에서 비행기가 지연될 확률이 매우 높습니다. (XGBoost 위험 패턴 감지)")
            elif delay_prob > 0.4:
                st.warning("주의: 약간의 지연 가능성이 존재합니다.")
            else:
                st.success("정상 운항 예상: 정시 출발 확률이 높습니다.")
                
            st.markdown("---")
            st.markdown(f"**선택하신 비행 조건:**\n* **출발:** {airport_names[origin]}\n* **도착:** {airport_names[destination]}\n* **항공사:** {airline_names[airline]}\n* **출발 시간:** {month}월 {hour}시\n* **비행 거리:** {distance} miles")

        with tab2:
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
                
                # 1. 사용자가 현재 선택한 6가지 조건만 필터링 (선택 조건 내에서의 중요도 비교)
                active_features = [
                    'Month', 'Distance', 'Hour', 
                    f"Airline_{airline}", f"Origin_Airport_{origin}", f"Destination_Airport_{destination}"
                ]
                imp_df = imp_df[imp_df['Feature'].isin(active_features)].copy()
                
                # 2. 그래프와 텍스트에 통일성 있게 출력할 '컬럼명(한글)' 포맷 생성 함수
                def get_kr_name(feat):
                    if feat == 'Month': return 'Month(운항 월)'
                    elif feat == 'Hour': return 'Hour(출발 시간대)'
                    elif feat == 'Distance': return 'Distance(비행 거리)'
                    elif feat.startswith('Airline_'):
                        code = feat.replace('Airline_', '')
                        return f"Airline({airline_names.get(code, code)})"
                    elif feat.startswith('Origin_Airport_'):
                        code = feat.replace('Origin_Airport_', '')
                        return f"Origin_Airport({airport_names.get(code, code)} 출발)"
                    elif feat.startswith('Destination_Airport_'):
                        code = feat.replace('Destination_Airport_', '')
                        return f"Destination_Airport({airport_names.get(code, code)} 도착)"
                    return feat

                imp_df['Kr_Name'] = imp_df['Feature'].apply(get_kr_name)
                imp_df = imp_df.sort_values(by='Importance', ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.barplot(x='Importance', y='Kr_Name', data=imp_df, palette='Reds_r' if delay_prob > 0.5 else 'Blues_r', ax=ax)
                ax.set_title("입력하신 비행 조건 중 지연 예측에 가장 큰 영향을 미친 요인", pad=15)
                ax.set_xlabel("영향도 (Feature Importance)")
                ax.set_ylabel("")
                
                st.pyplot(fig)
                
                top_kr_name = imp_df.iloc[0]['Kr_Name']
                    
                st.info(f"💡 **AI 해석 결과 요약:** 대표님께서 입력하신 조건들 중에서, 인공지능 모델이 지연 여부를 평가할 때 가장 강력하게 참고한 핵심 요인은 **[{top_kr_name}]** 입니다. 위 그래프에서 막대가 길수록 예측 결과에 결정적인 영향을 미쳤음을 의미합니다.")
            except Exception as e:
                st.info("시각화 데이터를 불러오는 중 오류가 발생했습니다.")
                
        with tab3:
            st.subheader("모델 신뢰도 및 성능 정보")
            st.markdown("본 인공지능 모델은 DACON 항공편 데이터 약 25만 건을 바탕으로 **XGBoost** 알고리즘을 사용해 학습되었습니다. 극심한 불균형 데이터를 해결하기 위해 `scale_pos_weight` 가중치를 튜닝하여, 실제 지연 비행기를 감지해내는 **재현율(Recall)을 65%까지 향상**시켰습니다.")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="모델 정확도 (Accuracy)", value="81.5%")
            with col2:
                st.metric(label="지연 감지율 (Recall)", value="65.0%")
                
            try:
                st.image("cm_after.png", caption="[학습 결과] XGBoost 튜닝 후 오차 행렬 (실제 지연 데이터를 효과적으로 감지)")
            except:
                pass
    else:
        st.info("좌측 사이드바에서 비행 조건을 설정한 후 '지연 확률 예측하기' 버튼을 눌러주세요.")
