# microgrid_app.py  (덮어쓰기용)
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="마이크로그리드 컨트롤 센터", layout="wide")
st.markdown("""<style>
.stApp { background-color: #07101a; color: #e6eef8; }
.title-font { font-size:22px !important; color:#ffffff; font-weight:600;}
.big-font { font-size:16px !important; color:#dbeafe;}
.card {background-color:#0f1726; padding:10px; border-radius:8px;}
.muted { color:#a6b8c8; font-size:13px }
</style>""", unsafe_allow_html=True)

# 세션 초기화
if 'blackout_log' not in st.session_state:
    st.session_state.blackout_log = []

# 사이드바: 기간/타임스텝 선택 추가
st.sidebar.header("시뮬레이션 설정")
period_choice = st.sidebar.selectbox("시뮬레이션 기간", ["24시간", "7일(168시간)", "30일(720시간)"])
period_map = {"24시간":24, "7일(168시간)":168, "30일(720시간)":720}
total_hours = period_map[period_choice]

# 타임스텝: 과거 방식처럼 1시간 단위 기본, 필요하면 30분/15분 선택 가능
time_step_min = st.sidebar.selectbox("타임스텝 (분)", [60, 30, 15], index=0)
dt_hours = time_step_min / 60.0
# 시간 스텝으로 계산되는 스텝 수
num_steps = int(total_hours * 60 / time_step_min)

# 날씨 / 발전 설정 (간단)
with st.sidebar.expander("날씨·발전 설정", expanded=True):
    weather_preset = st.selectbox("날씨 프리셋", ["맑음","흐림","비","폭우","태풍"])
    wind_speed = st.slider("평균 풍속 (m/s)", 0.0, 40.0, 8.0)
    solar_capacity_factor = st.slider("태양광 최대 계수 (0~1)", 0.0, 1.0, 0.7)
    wind_capacity_factor  = st.slider("풍력 최대 계수 (0~1)", 0.0, 1.0, 0.5)
    hydro_level = st.slider("저수지 수위 (0~1)", 0.0, 1.0, 0.6)

with st.sidebar.expander("수요·보조장치", expanded=True):
    demand_scale = st.slider("수요 배율", 0.5, 2.0, 1.0)
    smr_enabled = st.checkbox("SMR 사용", value=False)
    smr_output_fraction = st.slider("SMR 출력 비율", 0.0, 1.0, 0.5)

with st.sidebar.expander("배터리 설정", expanded=True):
    BATTERY_CAPACITY_KWH = st.number_input("배터리 용량 (kWh)", value=200.0)
    BATTERY_POWER_KW = st.number_input("배터리 최대 충/방전 전력 (kW)", value=100.0)
    init_soc_pct = st.slider("초기 SOC (%)", 0, 100, 50)
    CHARGE_EFF = st.slider("충전 효율", 0.5, 1.0, 0.95)
    DISCHARGE_EFF = st.slider("방전 효율", 0.5, 1.0, 0.95)

# 작은 옵션
random_variation = st.sidebar.checkbox("발전량 랜덤 변동 포함", True)

# 기본 발전 용량 (예시)
SOLAR_CAPACITY_KW = 100.0
WIND_CAPACITY_KW  = 120.0
HYDRO_CAPACITY_KW = 80.0
SMR_MAX_KW        = 200.0

# 시간 인덱스 (dt에 맞춰 생성)
start_time = datetime.now()
time_index = [start_time + timedelta(minutes=i*time_step_min) for i in range(num_steps)]
t = np.arange(num_steps)

# 날씨 프리셋 간단 보정
if weather_preset == "맑음":
    solar_weather_factor = 1.0; wind_weather_factor = 1.0; hydro_weather_factor = 0.95
elif weather_preset == "흐림":
    solar_weather_factor = 0.6; wind_weather_factor = 1.0; hydro_weather_factor = 1.0
elif weather_preset == "비":
    solar_weather_factor = 0.35; wind_weather_factor = 1.05; hydro_weather_factor = 1.1
elif weather_preset == "폭우":
    solar_weather_factor = 0.2; wind_weather_factor = 1.15; hydro_weather_factor = 1.3
elif weather_preset == "태풍":
    wind_speed = max(wind_speed, 25.0)
    solar_weather_factor = 0.15; wind_weather_factor = 0.95; hydro_weather_factor = 1.2
else:
    solar_weather_factor = wind_weather_factor = hydro_weather_factor = 1.0

# 일사 패턴 (시간해상도 맞춤)
phase = np.linspace(-np.pi/2, 3*np.pi/2, num_steps)
daylight = np.clip(np.sin(phase), 0, None)
solar_kw_inst = SOLAR_CAPACITY_KW * solar_capacity_factor * solar_weather_factor * daylight

# 풍력: 시간 프로파일, 속도->출력(간단)
np.random.seed(42)
wind_speed_profile = np.clip(np.random.normal(loc=wind_speed, scale=2.0, size=num_steps), 0.0, 50.0)
def wind_power_from_speed(speed_m_s, capacity_kw, factor):
    out = np.zeros_like(speed_m_s)
    for i, v in enumerate(speed_m_s):
        if v < 3.0:
            out[i] = 0.0
        elif v < 12.0:
            out[i] = capacity_kw * ((v - 3.0) / 9.0) * factor
        elif v <= 30.0:
            out[i] = capacity_kw * factor
        else:
            out[i] = 0.0
    return out
wind_kw_inst = wind_power_from_speed(wind_speed_profile, WIND_CAPACITY_KW, wind_capacity_factor) * wind_weather_factor

# 수력 (안정적)
hydro_kw_inst = HYDRO_CAPACITY_KW * hydro_level * hydro_weather_factor * (0.85 + 0.3 * np.random.rand(num_steps))

if random_variation:
    solar_kw_inst *= (1.0 + np.random.normal(0, 0.03, size=num_steps))
    wind_kw_inst  *= (1.0 + np.random.normal(0, 0.05, size=num_steps))
    hydro_kw_inst *= (1.0 + np.random.normal(0, 0.02, size=num_steps))

# SMR
smr_kw_inst = np.zeros(num_steps)
if smr_enabled:
    smr_kw_inst[:] = SMR_MAX_KW * smr_output_fraction

# 수요 (시간해상도에 맞춤)
demand_kw_inst = (80 + 40 * np.sin(np.linspace(0, 5, num_steps))) * demand_scale

# 배터리 시뮬레이션 — **중요 변경**: 모든 에너지는 kWh 단위로 dt를 곱함
battery_soc = np.zeros(num_steps)        # kWh
battery_power = np.zeros(num_steps)      # kW (+충전, -방전)
battery_soc[0] = (init_soc_pct/100.0) * BATTERY_CAPACITY_KWH

for i in range(num_steps):
    prev_soc = battery_soc[i-1] if i>0 else battery_soc[0]
    gen_kw = solar_kw_inst[i] + wind_kw_inst[i] + hydro_kw_inst[i] + smr_kw_inst[i]
    demand_kw = demand_kw_inst[i]
    surplus_kw = gen_kw - demand_kw   # kW at this timestep

    # 자동 ESS(간단 정책) — surplus 있으면 충전, 없으면 방전
    if surplus_kw > 0:
        # 가능한 충전 전력은 surplus과 배터리 전력 한계 중 최소
        charge_power_kw = min(surplus_kw, BATTERY_POWER_KW)
        # 실제로 가능한 충전 에너지 (kWh) = power * dt * 효율
        energy_can_add_kwh = charge_power_kw * dt_hours * CHARGE_EFF
        available_space_kwh = BATTERY_CAPACITY_KWH - prev_soc
        energy_added = min(energy_can_add_kwh, max(0.0, available_space_kwh))
        # 실제 충전 전력으로 환산 (kW) — energy_added / dt / eff_inv
        if dt_hours > 0:
            battery_power[i] = (energy_added / dt_hours) / CHARGE_EFF
        else:
            battery_power[i] = 0.0
        battery_soc[i] = prev_soc + energy_added
    else:
        # 방전: 필요한 방전 전력 = min( 부족량, 배터리 전력한계)
        needed_kw = min(-surplus_kw, BATTERY_POWER_KW)
        # 배터리에서 꺼낼 수 있는 에너지 (kWh) = prev_soc * 방전효율
        max_discharge_energy_kwh = prev_soc * DISCHARGE_EFF
        possible_discharge_energy_kwh = min(max_discharge_energy_kwh, needed_kw * dt_hours)
        # 실제 방전 전력 (kW)
        if dt_hours > 0:
            battery_power[i] = - (possible_discharge_energy_kwh / dt_hours) / DISCHARGE_EFF
        else:
            battery_power[i] = 0.0
        battery_soc[i] = prev_soc - (possible_discharge_energy_kwh / DISCHARGE_EFF)

    # 안전클램프(초과치 방지)
    battery_soc[i] = float(np.clip(battery_soc[i], 0.0, BATTERY_CAPACITY_KWH))

# 공급 계산: 배터리 방전(음수)이면 공급으로 더함, 충전이면 공급에서 차감(실제 네트워크로 안감)
supply_kw_inst = solar_kw_inst + wind_kw_inst + hydro_kw_inst + smr_kw_inst - np.maximum(battery_power, 0.0) + np.maximum(-battery_power, 0.0)

# 블랙아웃 판정 (공급 < 수요)
blackout_mask = supply_kw_inst < demand_kw_inst
blackout_count = int(np.sum(blackout_mask))
st.session_state.blackout_log = [time_index[i] for i in range(num_steps) if blackout_mask[i]]

# (이하 그래프/출력 부분은 이전 버전 레이아웃과 색상 유지)
COL_SOLAR="#FFD166"; COL_WIND="#06D6A0"; COL_HYDRO="#118AB2"; COL_SMR="#EF476F"; COL_SUPPLY="#ffffff"; COL_DEMAND="#FF4B4B"
st.markdown("<div class='title-font'>마이크로그리드 시뮬레이션 (타임스텝 반영)</div>", unsafe_allow_html=True)
st.markdown(f"<div class='muted'>타임스텝: {time_step_min}분 | 스텝수: {num_steps} | 최종 SOC: {battery_soc[-1]:.1f} kWh ({100.0*battery_soc[-1]/BATTERY_CAPACITY_KWH:.1f}%)</div>", unsafe_allow_html=True)

# 간단한 발전현황 그래프 (선 그래프)
fig = go.Figure()
fig.add_trace(go.Scatter(x=time_index, y=solar_kw_inst, mode='lines', name='태양광', line=dict(color=COL_SOLAR)))
fig.add_trace(go.Scatter(x=time_index, y=wind_kw_inst, mode='lines', name='풍력', line=dict(color=COL_WIND)))
fig.add_trace(go.Scatter(x=time_index, y=hydro_kw_inst, mode='lines', name='수력', line=dict(color=COL_HYDRO)))
if smr_enabled:
    fig.add_trace(go.Scatter(x=time_index, y=smr_kw_inst, mode='lines', name='SMR', line=dict(color=COL_SMR)))
fig.add_trace(go.Scatter(x=time_index, y=supply_kw_inst, mode='lines', name='총공급', line=dict(color=COL_SUPPLY, width=3)))
fig.add_trace(go.Scatter(x=time_index, y=demand_kw_inst, mode='lines', name='수요', line=dict(color=COL_DEMAND, width=3)))

blackout_times = [time_index[i] for i in range(num_steps) if blackout_mask[i]]
blackout_vals  = [supply_kw_inst[i] for i in range(num_steps) if blackout_mask[i]]
if blackout_times:
    fig.add_trace(go.Scatter(x=blackout_times, y=blackout_vals, mode='markers', marker=dict(color='#FF0000', size=8), name='블랙아웃'))

fig.update_layout(paper_bgcolor='#07101a', plot_bgcolor='#07101a', font_color='#e6eef8', yaxis_title="kW")
st.plotly_chart(fig, use_container_width=True, height=450)
