import streamlit as st
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import sys
import platform

# ✅ 紧急修复：NumPy 1.24+ 移除了 np.int
if not hasattr(np, 'int'):
    np.int = np.int64
    np.float = np.float64
    np.bool = np.bool_

# ✅ 第一步：必须是第一个 st 命令！
st.set_page_config(page_title="KOA 患者衰弱风险预测", layout="centered")

# ✅ 调试信息
with st.expander("🔧 调试信息", expanded=False):
    st.write(f"**Python版本**: `{sys.version.split()[0]}`")
    st.write(f"**系统环境**: `{platform.platform()}`")
    st.write(f"**numpy版本**: `{np.__version__}`")
    st.write(f"**xgboost版本**: `{xgb.__version__}`")

# ✅ 主标题
st.title("🩺 膝骨关节炎患者衰弱风险预测系统")
st.markdown("根据输入的临床特征，预测膝关节骨关节炎（KOA）患者发生衰弱（Frailty）的概率，并可视化决策依据。")

# ✅ 自定义CSS居中
st.markdown(
    """
    <style>
    .main > div {
        max-width: 800px;
        padding-left: 5rem;
        padding-right: 5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ✅ 加载模型和特征
@st.cache_resource
def load_model_and_features():
    model = xgb.Booster()
    model.load_model("xgb_model.json")
    with open("feature_names.pkl", 'rb') as f:
        feature_names = pickle.load(f)
    return model, feature_names

model, feature_names = load_model_and_features()

# ✅ 初始化SHAP解释器
@st.cache_resource
def create_explainer(_model):
    return shap.TreeExplainer(_model)

explainer = create_explainer(model)

# ✅ 创建输入表单
with st.form("patient_input_form"):
    st.markdown("---")
    st.subheader("📋 请填写以下信息")

    gender = st.radio("您的性别", ["女", "男"])
    age = st.number_input("您的年龄（岁）", min_value=0, max_value=120, value=60)
    smoking = st.radio("您是否吸烟？", ["否", "是"])
    bmi = st.number_input("输入您的 BMI（体重指数，kg/m²）", min_value=10.0, max_value=50.0, value=24.0, step=0.1)
    fall = st.radio("您过去一年是否发生过跌倒？", ["否", "是"])
    activity = st.radio("您觉得平时的体力活动水平", ["低水平", "中水平", "高水平"])
    complication = st.radio("您是否有并发症？", ["没有", "1个", "至少2个"])
    daily_activity = st.radio("您日常生活能力受限吗？", ["无限制", "有限制"])
    walk_speed = st.radio("输入您步行4m的速度（m/s）", ["小于1m/s", "大于等于1m/s"])
    sit_stand = st.radio("输入您连续5次坐立的时间（s）", ["小于12s", "大于等于12s"])
    platelet = st.number_input("输入您的血小板（×10^9/L）", min_value=0, max_value=1000, value=200)
    crea = st.number_input("输入您的crea（血肌酐，μmol/L）", min_value=0.0, max_value=1000.0, value=70.0, step=0.1)
    cysc = st.number_input("输入您的 CysC（胱抑素 C，mg/L）", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    wbc = st.number_input("输入您的wbc（白细胞，10^9/L）", min_value=0.0, max_value=50.0, value=6.0, step=0.1)

    submitted = st.form_submit_button("开始评估")

# ✅ 提交后处理预测（只写一次，且内容必须缩进！）
if submitted:
    with st.spinner('正在计算...'):
        # 构造输入数据
        input_data = {
            'gender': 1 if gender == "女" else 0,
            'age': age,
            'smoking': 1 if smoking == "是" else 0,
            'bmi': bmi,
            'fall': 1 if fall == "是" else 0,
            'PA_high': 1 if activity == "高水平" else 0,
            'PA_medium': 1 if activity == "中水平" else 0,
            'PA_low': 1 if activity == "低水平" else 0,
            'Complications_0': 1 if complication == "没有" else 0,
            'Complications_1': 1 if complication == "1个" else 0,
            'Complications_2': 1 if complication == "至少2个" else 0,
            'ADL': 1 if daily_activity == "有限制" else 0,
            'Walking_speed': 1 if walk_speed == "大于等于1m/s" else 0,
            'FTSST': 1 if sit_stand == "大于等于12s" else 0,
            'bl_plt': platelet,
            'bl_crea': crea,
            'bl_cysc': cysc,
            'bl_wbc': wbc
        }

        # 转为DataFrame并确保列顺序
        input_df = pd.DataFrame([input_data])
        for feature in feature_names:
            if feature not in input_df.columns:
                input_df[feature] = 0
        input_df = input_df[feature_names]

        # 预测
        dmatrix = xgb.DMatrix(input_df)
        pred_logodds = model.predict(dmatrix)[0]
        frail_prob = 1 / (1 + np.exp(-pred_logodds))
        pred_label = 1 if frail_prob > 0.5 else 0

        # 显示结果
        st.success(f"📊 预测结果: 患者衰弱概率为 {frail_prob * 100:.2f}%")

        if frail_prob > 0.8:
            st.error("""⚠️ **高风险：建议立即临床干预**""")
            st.write("- 每周随访监测")
            st.write("- 必须物理治疗干预")
            st.write("- 全面评估并发症")
        elif frail_prob > 0.3:
            st.warning("""⚠️ **中风险：建议定期监测**""")
            st.write("- 每3-6个月评估一次")
            st.write("- 建议适度运动计划")
            st.write("- 基础营养评估")
        else:
            st.success("""✅ **低风险：建议常规健康管理**""")
            st.write("- 每年体检一次")
            st.write("- 保持健康生活方式")
            st.write("- 预防性健康指导")

        # SHAP可视化（使用临时文件保存图像，避免Streamlit中force_plot空白问题）
        try:
            shap_values = explainer.shap_values(dmatrix)
            expected_value = explainer.expected_value
            if isinstance(expected_value, np.ndarray):
                expected_value = expected_value[1] if pred_label == 1 else expected_value[0]
            if isinstance(shap_values, list):
                shap_value = shap_values[1][0] if pred_label == 1 else shap_values[0][0]
            else:
                shap_value = shap_values[0]

            # 特征名称映射
            feature_names_mapping = {
                'age': f'Age={int(age)}',
                'bmi': f'BMI={bmi:.1f}',
                'bl_wbc': f'Wbc={wbc:.1f}',
                'bl_crea': f'Crea={crea:.1f}',
                'bl_plt': f'Plt={platelet}',
                'bl_cysc': f'Cysc={cysc:.1f}',
                'Complications_0': f'Complications={"无" if complication=="没有" else "有"}',
                'Complications_1': f'Complications={"有" if complication=="1个" else "无"}',
                'Complications_2': f'Complications={"≥2" if complication=="至少2个" else "无"}',
                'FTSST': f'FTSST={"≥12s" if sit_stand=="大于等于12s" else "<12s"}',
                'Walking_speed': f'Walk={"≥1m/s" if walk_speed=="大于等于1m/s" else "<1m/s"}',
                'fall': f'Fall={"是" if fall=="是" else "否"}',
                'ADL': f'ADL={"受限" if daily_activity=="有限制" else "正常"}',
                'gender': f'Gender={"女" if gender=="女" else "男"}',
                'PA_high': f'PA={"高" if activity=="高水平" else "中/低"}',
                'PA_medium': f'PA={"中" if activity=="中水平" else "高/低"}',
                'PA_low': f'PA={"低" if activity=="低水平" else "高/中"}',
                'smoking': f'Smoke={"是" if smoking=="是" else "否"}'
            }

            st.subheader(f"🧠 决策依据分析（{'衰弱' if pred_label == 1 else '非衰弱'}类）")

            # 清除之前的图
            plt.close('all')

            # 创建 force_plot 图像
            fig = shap.force_plot(
                base_value=expected_value,
                shap_values=shap_value,
                features=input_df.iloc[0],
                feature_names=[feature_names_mapping.get(f, f) for f in input_df.columns],
                matplotlib=True,
                show=False,
                plot_cmap="RdBu"
            )

            # 使用 tempfile 保存图像
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                # 保存图像
                fig.savefig(tmpfile.name, bbox_inches='tight', dpi=300, facecolor='white')
                # 在 Streamlit 中显示
                st.image(tmpfile.name, use_column_width=True)

            # 清理 SHAP 图像（避免内存泄漏）
            plt.close(fig)

        except Exception as e:
            st.error(f"SHAP可视化失败: {str(e)}")
            st.warning("建议刷新页面或检查输入数据。")

        st.markdown("""
        **图例说明:**
        - 🔴 **红色**：增加衰弱风险的特征  
        - 🟢 **绿色**：降低衰弱风险的特征  
        """)

# ✅ 页脚
st.markdown("---")
st.caption("©2025 KOA预测系统 | 仅供临床参考")



