import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="하이브리드 마이크로그리드 시뮬레이션", layout="wide")
st.title("하이브리드 마이크로그리드 시뮬레이션 (마을 kW + SMR MW)")

st.sidebar.header("시뮬레이션 설정")
hours = st.sidebar.slider("시뮬레이션 시간(시간)", 1, 168, 24)
step_min = st.sidebar.selectbox("타임스텝(분)", [5,10,15,30,60], 2)
steps = int(hours * 60 / step_min)

st.sidebar.subheader("재생에너지 정격출력(kW)")
pv_cap  = st.sidebar.number_input("태양광(kW)", 0, 5000, 500)
wind_cap = st.sidebar.number_input("풍력(kW)", 0, 5000, 1200)
hydro_cap = st.sidebar.number_input("수력(kW)", 0, 5000, 800)

st.sidebar.subheader("SMR (보조 전력, MW)")
smr_mw = st.sidebar.number_input("SMR 최대 출력(MW)", 0.0, 10.0, 1.2, 0.1)
smr_cap = smr_mw * 1000
smr_base = st.sidebar.number_input("SMR 기본 출력(kW)", 0, 2000, 100)
smr_ramp = st.sidebar.number_input("SMR 출력변화 제한(kW/분)", 10, 2000, 400)

st.sidebar.subheader("배터리 시스템")
batt_cap = st.sidebar.number_input("배터리 용량(kWh)", 0, 50000, 3000)
init_soc = st.sidebar.slider("초기 배터리 충전 상태(kWh)", 0, int(batt_cap if batt_cap > 0 else 1), int(min(3000, batt_cap)))
batt_chg = st.sidebar.number_input("충전 최대(kW)", 0, 5000, 600)
batt_dch = st.sidebar.number_input("방전 최대(kW)", 0, 5000, 600)
batt_eff = st.sidebar.slider("효율(%)", 50, 100, 90) / 100

st.sidebar.subheader("전력 수요")
profile = st.sidebar.selectbox("수요 유형", ["마을","학교","병원","데이터센터"])
base_load = st.sidebar.number_input("기본 수요(kW)", 100, 8000, 1400)
peak_add  = st.sidebar.number_input("피크 추가(kW)", 0, 6000, 900)

st.sidebar.subheader("날씨")
weather = st.sidebar.selectbox("날씨", ["맑음","보통","흐림","비","태풍"])
sf = st.sidebar.slider("일조 강도", 0.0,1.0,0.6)
wf = st.sidebar.slider("풍력 강도", 0.0,1.0,0.6)
hf = st.sidebar.slider("수력 강도", 0.0,1.0,0.5)

st.sidebar.subheader("블랙아웃 제어")
allow_shed = st.sidebar.checkbox("부하 차단 허용", True)
shed_p = st.sidebar.slider("최대 부하 차단(%)", 0,50,20)

start = datetime.now()
time = [start + timedelta(minutes=i*step_min) for i in range(steps)]

df = pd.DataFrame({"t": time})
df["pv"] = pv_cap * sf * (np.sin(np.linspace(0, 3.14, steps)) ** 2)
df["wind"] = wind_cap * wf * (np.random.rand(steps))
df["hydro"] = hydro_cap * hf * (np.random.rand(steps))

if weather == "태풍":
    df["wind"] = 0

df["renew"] = df["pv"] + df["wind"] + df["hydro"]
df["load"] = base_load + (np.random.rand(steps)*peak_add)

df["smr"] = smr_base
df["sup"] = df["renew"] + df["smr"]
df["deficit"] = df["load"] - df["sup"]

df["black"] = df["deficit"] > 0

st.subheader("발전 및 수요 그래프")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.t, y=df.pv, name="태양광"))
fig.add_trace(go.Scatter(x=df.t, y=df.wind, name="풍력"))
fig.add_trace(go.Scatter(x=df.t, y=df.hydro, name="수력"))
fig.add_trace(go.Scatter(x=df.t, y=df.load, name="수요", line=dict(width=3, dash="dot")))
fig.update_layout(template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

st.subheader("블랙아웃 발생 횟수")
st.metric("정전 횟수", df["black"].sum())
