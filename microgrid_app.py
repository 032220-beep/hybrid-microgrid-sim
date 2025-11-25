# microgrid_app.py
# 하이브리드 마이크로그리드 시뮬레이터 (최신: 자동/버튼 실행 + 블랙아웃 시점/지속 표 추가)
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="하이브리드 마이크로그리드 시뮬레이터", layout="wide")
st.title("하이브리드 마이크로그리드 시뮬레이션")

# -----------------------
# 사이드바: UI / 설정
# -----------------------
st.sidebar.header("시뮬레이션 설정")

# 시뮬레이션 기간 (시간) — 24 ~ 8760
sim_hours = st.sidebar.number_input(
    "시뮬레이션 시간 (시간)", min_value=24, max_value=8760, value=24, step=1,
    help="24시간 ~ 8760시간(연간)"
)

# 타임스텝 (분) — 15/30/60/180
time_step_min = st.sidebar.selectbox(
    "타임스텝 (분)", options=[15, 30, 60, 180], index=2, help="시뮬레이션 시간 해상도"
)

# 랜덤 시드
seed = st.sidebar.number_input("랜덤 시드", min_value=0, max_value=999_999, value=42)

# 장소(수요 프로필)
st.sidebar.subheader("장소 / 수요 프로필")
profile = st.sidebar.selectbox(
    "장소 선택",
    options=["마을", "학교", "병원", "데이터센터", "공장(산업)", "주거지역"],
    index=0
)

# 재생정격(단위: kW)
st.sidebar.subheader("재생에너지 정격출력 (kW)")
pv_cap = st.sidebar.number_input("태양광 (kW)", min_value=0, max_value=20000, value=500)
wind_cap = st.sidebar.number_input("풍력 (kW)", min_value=0, max_value=20000, value=1200)
hydro_cap = st.sidebar.number_input("수력 (kW)", min_value=0, max_value=20000, value=800)

# SMR (입력 MW)
st.sidebar.subheader("SMR (설정)")
smr_mw = st.sidebar.number_input("SMR 최대 출력 (MW)", min_value=0.0, max_value=20.0, value=1.2, step=0.1)
smr_cap = int(round(smr_mw * 1000))  # kW
smr_base = st.sidebar.number_input("SMR 기본 출력 (kW)", min_value=0, max_value=smr_cap, value=min(100, smr_cap))
smr_ramp = st.sidebar.number_input("SMR 출력변화 제한 (kW/분)", min_value=0, max_value=5000, value=400)

# 배터리 (ESS)
st.sidebar.subheader("ESS 배터리")
batt_cap = st.sidebar.number_input("배터리 용량 (kWh)", min_value=0, max_value=100000, value=3000)
batt_chg = st.sidebar.number_input("충전 최대 (kW)", min_value=0, max_value=5000, value=600)
batt_dch = st.sidebar.number_input("방전 최대 (kW)", min_value=0, max_value=5000, value=600)
batt_eff_percent = st.sidebar.slider("배터리 효율 (%)", min_value=50, max_value=100, value=90)
batt_eff = batt_eff_percent / 100.0

# 기후/날씨
st.sidebar.subheader("날씨")
weather = st.sidebar.selectbox("날씨", options=["맑음", "보통", "흐림", "비", "태풍"], index=1)
sun_strength = st.sidebar.slider("일조 강도 (0-1)", min_value=0.0, max_value=1.0, value=0.6)
wind_strength = st.sidebar.slider("풍력 강도 (0-1)", min_value=0.0, max_value=1.0, value=0.5)
hydro_strength = st.sidebar.slider("수력(유입) 강도 (0-1)", min_value=0.0, max_value=1.0, value=0.4)

# 정전/부하 차단 정책
st.sidebar.subheader("정전 방지 / 안전장치")
allow_shed = st.sidebar.checkbox("부하 차단 허용", value=True)
shed_percent = st.sidebar.slider("최대 부하 차단 (%)", 0, 100, 20)
reserve_margin = st.sidebar.slider("예비율 (kW)", 0, 1000, 50)  # (모델 내 필수 사용 아님, 필요시 확장용)

# 비용·탄소 (원/kWh, gCO2/kWh)
st.sidebar.subheader("비용 & 탄소")
c_pv = st.sidebar.number_input("태양광 비용 (원/kWh)", min_value=0.0, max_value=1000.0, value=50.0)
c_wd = st.sidebar.number_input("풍력 비용 (원/kWh)", min_value=0.0, max_value=1000.0, value=40.0)
c_hd = st.sidebar.number_input("수력 비용 (원/kWh)", min_value=0.0, max_value=1000.0, value=30.0)
c_smr = st.sidebar.number_input("SMR 비용 (원/kWh)", min_value=0.0, max_value=2000.0, value=65.0)
e_smr = st.sidebar.number_input("SMR 탄소 배출 (gCO₂/kWh)", min_value=0.0, max_value=1000.0, value=12.0)

# SMR 운전 모드
smr_mode = st.sidebar.selectbox(
    "SMR 운전 모드", options=["자동(수요맞춤)", "항상기본(베이스만)", "우선충전(피크차단 우선)"], index=0
)

# 실행 방법: 자동/버튼
st.sidebar.markdown("---")
run_mode = st.sidebar.radio("실행 모드", ["버튼 실행", "자동 실행"], index=0, horizontal=True)
run_button = st.sidebar.button("시뮬레이션 실행")
st.sidebar.caption("시뮬레이션 실행 시 현재 설정으로 결과를 계산합니다.")
should_run = (run_mode == "자동 실행") or run_button

# -----------------------
# 내부 변수
# -----------------------
np.random.seed(int(seed))
steps = int(sim_hours * 60 / time_step_min)
start_time = datetime.now()

# 수요 모델(장소별 기본치)
def demand_profile_base(profile_name: str) -> float:
    mapping = {
        "마을": 1400,
        "학교": 700,
        "병원": 2000,
        "데이터센터": 3000,
        "공장(산업)": 2500,
        "주거지역": 900,
    }
    return mapping.get(profile_name, 1400)

base_load = demand_profile_base(profile)

# 시간별 수요 함수
def demand_at(i: int) -> float:
    d_period = int(24 * 60 / time_step_min)
    x = (i % d_period) / d_period
    daily_cycle = 1.0 + 0.2 * np.sin(2 * np.pi * (x - 0.15)) + 0.08 * np.cos(4 * np.pi * x)
    hour = (start_time + timedelta(minutes=i * time_step_min)).hour
    ev_factor = 1.3 if 18 <= hour <= 22 else 1.0
    spike = base_load * 0.5 * np.random.rand() if np.random.rand() < 0.01 else 0.0
    noise = np.random.normal(0, base_load * 0.03)
    profile_mult = {
        "마을": 1.0, "학교": 0.6, "병원": 1.6,
        "데이터센터": 2.1, "공장(산업)": 1.8, "주거지역": 0.95
    }[profile]
    return max(0.0, profile_mult * (base_load * daily_cycle * ev_factor + spike) + noise)

# 발전원 모형
def solar_profile(i: int) -> float:
    d_period = int(24 * 60 / time_step_min)
    pos = (i % d_period) / d_period
    shape = np.exp(-((pos - 0.5) ** 2) * 20)  # 정오 피크
    weather_factor = {"맑음": 1.0, "보통": 0.85, "흐림": 0.6, "비": 0.4, "태풍": 0.15}[weather]
    return max(0.0, shape * sun_strength * weather_factor + np.random.normal(0, 0.02))

def wind_profile(i: int) -> float:
    if weather == "태풍" and wind_strength > 0.7 and np.random.rand() < 0.6:
        return 0.0
    base = 0.3 + 0.7 * wind_strength + 0.2 * np.sin(i / 45.0)
    weather_adj = {"맑음": 0.9, "보통": 1.0, "흐림": 1.05, "비": 1.1, "태풍": 1.2}[weather]
    return max(0.0, base * weather_adj + np.random.normal(0, 0.12))

def hydro_profile(i: int) -> float:
    r = 0.0
    if weather in ["비", "태풍"] and np.random.rand() < 0.05:
        r = np.random.uniform(0.15, 0.6)
    return max(0.0, 0.35 * hydro_strength + r + np.random.normal(0, 0.03))

# 시뮬레이션
def run_sim():
    time_index = [start_time + timedelta(minutes=i * time_step_min) for i in range(steps)]
    pv_out = np.zeros(steps)
    wd_out = np.zeros(steps)
    hd_out = np.zeros(steps)
    ld = np.zeros(steps)
    smr_out = np.zeros(steps)
    soc = np.zeros(steps)
    shed = np.zeros(steps)
    blackout = np.zeros(steps, dtype=bool)

    bat = batt_cap  # 초기 완충
    now_smr = smr_base
    total_cost = 0.0
    total_carbon = 0.0
    dt_h = time_step_min / 60.0

    for i in range(steps):
        p = pv_cap * solar_profile(i)
        w = wind_cap * wind_profile(i)
        h = hydro_cap * hydro_profile(i)
        d = demand_at(i)
        ld[i] = d

        ren = p + w + h
        net = ren - d  # +면 잉여, -면 부족

        # 배터리 충방전
        if net > 0:
            possible_chg = min(net, batt_chg)
            chg_kwh = possible_chg * dt_h
            avail_space = max(0.0, batt_cap - bat)
            chg_kwh = min(chg_kwh, avail_space)
            bat += chg_kwh * batt_eff
            net -= (chg_kwh / dt_h) if dt_h > 0 else 0.0
        if net < 0:
            needed = -net
            possible_dch = min(needed, batt_dch)
            dch_kwh = possible_dch * dt_h
            dch_kwh = min(dch_kwh, bat)
            bat -= dch_kwh
            supplied = dch_kwh * batt_eff / dt_h if dt_h > 0 else 0.0
            net += supplied

        # SMR 제어
        target = smr_base
        if smr_mode == "자동(수요맞춤)":
            if net < 0:
                need = -net
                headroom = max(0, smr_cap - smr_base)
                add = min(need, headroom)
                target = smr_base + add
        elif smr_mode == "항상기본(베이스만)":
            target = smr_base
        elif smr_mode == "우선충전(피크차단 우선)":
            target = min(smr_cap, smr_base + max(0.0, -net))

        # Ramp limit
        lim = smr_ramp * time_step_min
        diff = target - now_smr
        if abs(diff) > lim:
            diff = np.sign(diff) * lim
        now_smr += diff

        tot_supply = ren + now_smr
        deficit = d - tot_supply

        s_cut = 0.0
        black = False
        if deficit > 0:
            if allow_shed:
                cut_allowed = d * (shed_percent / 100.0)
                s_cut = min(deficit, cut_allowed)
                deficit -= s_cut
            if deficit > 0:
                black = True

        # 비용/탄소
        total_cost += (p * dt_h) * c_pv + (w * dt_h) * c_wd + (h * dt_h) * c_hd + (now_smr * dt_h) * c_smr
        total_carbon += (now_smr * dt_h) * e_smr

        pv_out[i] = p
        wd_out[i] = w
        hd_out[i] = h
        smr_out[i] = now_smr
        soc[i] = bat
        shed[i] = s_cut
        blackout[i] = black

    df = pd.DataFrame({
        "time": time_index,
        "pv": pv_out,
        "wind": wd_out,
        "hydro": hd_out,
        "smr": smr_out,
        "load": ld,
        "soc": soc,
        "shed": shed,
        "blackout": blackout.astype(int),
    })
    df["renewables"] = df["pv"] + df["wind"] + df["hydro"]
    df["supply"] = df["renewables"] + df["smr"]

    return df, total_cost, total_carbon

# 블랙아웃 구간 표 생성
def blackout_segments(df: pd.DataFrame, step_min: int) -> pd.DataFrame:
    segs = []
    in_seg = False
    start = None
    length = 0
    for i, b in enumerate(df.blackout):
        if b and not in_seg:
            in_seg = True
            start = df.time.iloc[i]
            length = 1
        elif b and in_seg:
            length += 1
        elif (not b) and in_seg:
            end = df.time.iloc[i]
            minutes = length * step_min
            segs.append((start, end, minutes))
            in_seg = False
            start = None
            length = 0
    if in_seg:
        end = df.time.iloc[len(df) - 1] + timedelta(minutes=step_min)
        minutes = length * step_min
        segs.append((start, end, minutes))
    if not segs:
        return pd.DataFrame(columns=["시작", "종료", "지속(분)", "지속(시:분)"])
    out = pd.DataFrame(segs, columns=["시작", "종료", "지속(분)"])
    out["지속(시:분)"] = out["지속(분)"].apply(lambda m: f"{int(m//60)}:{int(m%60):02d}")
    return out

# -----------------------
# 실행/결과
# -----------------------
if should_run:
    with st.spinner("시뮬레이션 실행 중..."):
        df, total_cost, total_carbon = run_sim()
    st.success("시뮬레이션 완료")

    tab1, tab2, tab3, tab4 = st.tabs(["발전 현황", "안정성", "비용 + 탄소", "분석 (도넛)"])

    # 발전 현황
    with tab1:
        fig = go.Figure()
        # 재생 스택 면적
        fig.add_trace(go.Scatter(x=df.time, y=df.pv, name="태양광", stackgroup="renew", mode="none"))
        fig.add_trace(go.Scatter(x=df.time, y=df.wind, name="풍력", stackgroup="renew", mode="none"))
        fig.add_trace(go.Scatter(x=df.time, y=df.hydro, name="수력", stackgroup="renew", mode="none"))
        # SMR 막대
        fig.add_trace(go.Bar(x=df.time, y=df.smr, name="SMR", marker=dict(color="#7fb3ff"), opacity=0.8))
        # 수요선
        fig.add_trace(go.Scatter(x=df.time, y=df.load, name="수요", line=dict(color="white", width=2), mode="lines"))
        # 블랙아웃 음영
        for i, b in enumerate(df.blackout):
            if b:
                start = df.time.iloc[i]
                end = start + timedelta(minutes=time_step_min)
                fig.add_vrect(x0=start, x1=end, fillcolor="red", opacity=0.25, line_width=0)

        fig.update_layout(
            template="plotly_dark", height=600,
            yaxis_title="출력 (kW)", xaxis_title="시간",
            legend=dict(orientation="h")
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"**정전(블랙아웃) 횟수:** {int(df.blackout.sum())}")

        # 블랙아웃 시점/지속 표
        bo_df = blackout_segments(df, time_step_min)
        with st.expander("블랙아웃 구간(시작/종료/지속 시간) 보기", expanded=False):
            if len(bo_df) == 0:
                st.info("시뮬레이션 구간에서 블랙아웃이 발생하지 않았습니다.")
            else:
                total_min = int(bo_df["지속(분)"].sum())
                st.write(f"총 블랙아웃 횟수: **{len(bo_df)}회**, 총 지속 시간: **{total_min}분 ({total_min//60}시간 {total_min%60}분)**")
                st.dataframe(bo_df, use_container_width=True)

    # 안정성
    with tab2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df.time, y=df.soc, name="배터리 SOC (kWh)", line=dict(color="#F7D060", width=3)))
        deficit_series = (df.load - df.supply).clip(lower=0)
        fig2.add_trace(go.Scatter(x=df.time, y=deficit_series, name="공급 부족 (kW)", line=dict(color="red", width=2)))
        for i, b in enumerate(df.blackout):
            if b:
                start = df.time.iloc[i]
                end = start + timedelta(minutes=time_step_min)
                fig2.add_vrect(x0=start, x1=end, fillcolor="red", opacity=0.25, line_width=0)
        fig2.update_layout(template="plotly_dark", height=450, yaxis_title="kW / kWh")
        st.plotly_chart(fig2, use_container_width=True)
        st.metric("정전(블랙아웃) 횟수", int(df.blackout.sum()))

    # 비용 + 탄소
    with tab3:
        dt_h = time_step_min / 60.0
        cum_cost = np.cumsum((df.pv * dt_h) * c_pv + (df.wind * dt_h) * c_wd + (df.hydro * dt_h) * c_hd + (df.smr * dt_h) * c_smr)
        cum_carbon = np.cumsum((df.smr * dt_h) * e_smr)
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=df.time, y=cum_cost, name="누적 비용 (원)", line=dict(color="#4B8BF5")))
        fig3.add_trace(go.Scatter(x=df.time, y=cum_carbon, name="누적 탄소 (gCO₂)", line=dict(color="#3DDC97"), yaxis="y2"))
        fig3.update_layout(
            template="plotly_dark", height=480,
            yaxis=dict(title="누적 비용 (원)"),
            yaxis2=dict(title="누적 탄소 (gCO₂)", overlaying="y", side="right")
        )
        st.plotly_chart(fig3, use_container_width=True)
        col1, col2 = st.columns(2)
        col1.metric("총 비용 (원)", f"{int(float(cum_cost[-1])):,}")
        col2.metric("총 탄소배출 (gCO₂)", f"{int(float(cum_carbon[-1])):,}")

    # 분석(도넛)
    with tab4:
        dt_h = time_step_min / 60.0
        total_pv = float((df.pv * dt_h).sum())
        total_wd = float((df.wind * dt_h).sum())
        total_hd = float((df.hydro * dt_h).sum())
        total_smr_kwh = float((df.smr * dt_h).sum())

        gen_values = [total_pv, total_wd, total_hd, total_smr_kwh]
        gen_labels = ["태양광", "풍력", "수력", "SMR"]

        cost_vals = [total_pv * c_pv, total_wd * c_wd, total_hd * c_hd, total_smr_kwh * c_smr]
        carbon_vals = [0.0, 0.0, 0.0, total_smr_kwh * e_smr]

        c1, c2, c3 = st.columns(3)
        fig_d1 = go.Figure(go.Pie(labels=gen_labels, values=gen_values, hole=0.5))
        fig_d1.update_traces(textinfo="percent+label")
        fig_d1.update_layout(title="발전원 비중 (kWh)", template="plotly_dark", height=360)
        c1.plotly_chart(fig_d1, use_container_width=True)

        fig_d2 = go.Figure(go.Pie(labels=gen_labels, values=cost_vals, hole=0.5))
        fig_d2.update_traces(textinfo="percent+label")
        fig_d2.update_layout(title="비용 분해 (원)", template="plotly_dark", height=360)
        c2.plotly_chart(fig_d2, use_container_width=True)

        fig_d3 = go.Figure(go.Pie(labels=gen_labels, values=carbon_vals, hole=0.5))
        fig_d3.update_traces(textinfo="percent+label")
        fig_d3.update_layout(title="탄소 배출 분해 (gCO₂)", template="plotly_dark", height=360)
        c3.plotly_chart(fig_d3, use_container_width=True)

        st.caption("각 도넛은 시뮬레이션 기간 동안 각 발전원별 kWh/비용/탄소 기여도를 보여줍니다.")

else:
    st.info("설정을 조정한 뒤 **시뮬레이션 실행**(또는 자동 실행 모드)으로 결과를 생성하세요.")
