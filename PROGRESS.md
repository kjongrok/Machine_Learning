# 🚗 중고차 가격 예측 서비스 진행 상황

## 1. 현재 진행 상태
- `app.py` 파일 내 코드를 모두 지우고 깔끔하게 초기화해 두었습니다. (다음 작업 시 백지 상태에서 처음부터 작성 시작)

## 2. 머신러닝 모델 분석 요약 (`김종록_머신러닝_프로젝트.ipynb`)
`app.py` 코드를 작성할 때 필요한 모델의 핵심 설계 정보입니다.

*   **학습 데이터셋:** `dataset/archive/car data.csv` 
    *   *주의사항: 현재 폴더에 있는 `car details v4.csv`는 컬럼 구조가 다르므로 Streamlit 앱 제작 시 참고하지 않도록 주의가 필요합니다.*
*   **타겟 변수 (예측 대상):** `Selling_Price`
*   **입력 변수 (Feature):** `Year`, `Present_Price`, `Kms_Driven`, `Owner`, `Car_Name`, `Fuel_Type`, `Seller_Type`, `Transmission`
*   **데이터 전처리 방식:** `pd.get_dummies(drop_first=True)`를 사용하여 다중공선성을 방지한 원핫 인코딩 적용.
*   **사용된 알고리즘:** 랜덤 포레스트 회귀 (`RandomForestRegressor`)
*   **산출물 (저장된 파일):** 
    - `car_price_model.pkl`: 학습 완료된 모델 파일
    - `model_columns.pkl`: 모델이 인식하는 원핫 인코딩된 최종 컬럼 리스트

## 3. 다음 작업 (To-Do)
- [ ] `app.py` 작성 시작
  - `model_columns.pkl`에 필요한 정보(`Year` 등 8개 변수)를 사용자로부터 입력받는 Streamlit UI(수치형, 선택형) 구현
  - 예측 버튼 클릭 시, 사용자가 선택한 범주형 데이터를 `model_columns` 구조에 맞추어 `1`과 `0`으로 변환(원핫 인코딩)하는 로직 작성
  - 모델을 이용해 최종 가격을 예측하고 결과를 화면에 띄우기
