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
import numpy as np
if not hasattr(np, 'int'):
    np.int = np.int64
    np.float = np.float64
    np.bool = np.bool_
# ✅ 第一步：必须是第一个 st 命令！
st.set_page_config(page_title="KOA 患者衰弱风险预测", layout="centered")

# ✅ 第二步：现在可以安全地使用 st.write 了
with st.expander("🔧 调试信息", expanded=False):
    st.write(f"**Python版本**: `{sys.version.split()[0]}`")
    st.write(f"**系统环境**: `{platform.platform()}`")
    st.write(f"**numpy版本**: `{np.__version__}`")
    st.write(f"**xgboost版本**: `{xgb.__version__}`")

# ✅ 第三步：继续你的主逻辑
st.title("🩺 膝骨关节炎患者衰弱风险预测系统")
st.markdown("根据输入的临床特征，预测膝关节骨关节炎（KOA）患者发生衰弱（Frailty）的概率，并可视化决策依据。")

# 自定义CSS实现全页面居中
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

# ✅ 加载模型和特征名称（关键修复在这里！）
@st.cache_resource
def load_model_and_features():
    # ✅ 模型路径：使用 XGBoost 原生 JSON 格式
    model_path = "xgb_model.json"
    feature_path = "feature_names.pkl"
    
    # ✅ 1. 使用 XGBoost 原生方式加载模型（不再用 pickle！）
    model = xgb.Booster()
    model.load_model(model_path)  # ✅ 正确方式加载 .json
    
    # ✅ 2. 用 pickle 加载特征名称（这个小文件一般没问题）
    with open(feature_path, 'rb') as f:
        feature_names = pickle.load(f)
    
    return model, feature_names

# ✅ 加载模型和特征
model, feature_names = load_model_and_features()

# ✅ 初始化SHAP解释器
@st.cache_resource
def create_explainer(_model):
    explainer = shap.TreeExplainer(_model)
    return explainer

explainer = create_explainer(model)

# 创建输入表单
with st.form("patient_input_form"):
    st.markdown("---")
    st.subheader("📋 请填写以下信息") 
    
    # 性别
    gender = st.radio("您的性别", ["女", "男"])
    
    # 年龄
    age = st.number_input("您的年龄（岁）", min_value=0, max_value=120, value=60)
    
    # 吸烟
    smoking = st.radio("您是否吸烟？", ["否", "是"])
    
    # BMI
    bmi = st.number_input("输入您的 BMI（体重指数，kg/m²）", min_value=10.0, max_value=50.0, value=24.0, step=0.1)
    
    # 跌倒
    fall = st.radio("您过去一年是否发生过跌倒？", ["否", "是"])
    
    # 体力活动水平
    activity = st.radio("您觉得平时的体力活动水平", ["低水平", "中水平", "高水平"])
    
    # 并发症
    complication = st.radio("您是否有并发症？", ["没有", "1个", "至少2个"])
    
    # 日常生活能力
    daily_activity = st.radio("您日常生活能力受限吗？", ["无限制", "有限制"])
    
    # 步行速度
    walk_speed = st.radio("输入您步行4m的速度（m/s）", ["小于1m/s", "大于等于1m/s"])
    
    # 坐立时间
    sit_stand = st.radio("输入您连续5次坐立的时间（s）", ["小于12s", "大于等于12s"])
    
    # 血小板
    platelet = st.number_input("输入您的血小板（×10^9/L）", min_value=0, max_value=1000, value=200)
    
    # 血肌酐
    crea = st.number_input("输入您的crea（血肌酐，μmol/L）", min_value=0.0, max_value=1000.0, value=70.0, step=0.1)
    
    # 胱抑素C
    cysc = st.number_input("输入您的 CysC（胱抑素 C，mg/L）", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    
    # 白细胞
    wbc = st.number_input("输入您的wbc（白细胞，10^9/L）", min_value=0.0, max_value=50.0, value=6.0, step=0.1)
    
    submitted = st.form_submit_button("开始评估")


if submitted:
    with st.spinner('正在计算...'):
        st.experimental_rerun()
# 处理输入数据并预测
if submitted:
    # 将输入转换为模型需要的格式
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
    
    # 创建DataFrame
    input_df = pd.DataFrame([input_data])
    
    # 确保所有特征都存在
    for feature in feature_names:
        if feature not in input_df.columns:
            input_df[feature] = 0
    
    # 重新排序列
    input_df = input_df[feature_names]
    
    # ✅ 进行预测：注意 model 是 Booster，没有 predict_proba
    # 我们用 predict + sigmoid 模拟 predict_proba
    dmatrix = xgb.DMatrix(input_df)
    pred_logodds = model.predict(dmatrix)[0]
    frail_prob = 1 / (1 + np.exp(-pred_logodds))  # sigmoid 转概率
    pred_label = 1 if frail_prob > 0.5 else 0
    
    # 显示预测结果
    st.success(f"📊 预测结果: 患者衰弱概率为 {frail_prob*100:.2f}%")
    
    # 调整后的风险等级提示
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
    
    # SHAP分析可视化
    try:
        # ✅ 计算SHAP值（注意：explainer 接收 DMatrix 或 numpy）
        shap_values = explainer.shap_values(dmatrix)
        
        # 获取当前类别的SHAP值
        expected_value = explainer.expected_value
        if isinstance(expected_value, np.ndarray):
            expected_value = expected_value[1] if pred_label == 1 else expected_value[0]
        if isinstance(shap_values, list):
            shap_value = shap_values[1][0] if pred_label == 1 else shap_values[0][0]
        else:
            shap_value = shap_values[0]
        
        # 创建特征名称映射（可简化）
        feature_names_mapping = {
            'age': f'Age={int(age)}',
            'bmi': f'BMI={bmi:.1f}',
            'bl_wbc': f'Wbc={wbc:.1f}',
            'bl_crea': f'Crea={crea:.1f}',
            'bl_plt': f'Plt={int(platelet)}',
            'bl_cysc': f'Cysc={cysc:.1f}',
            'Complications_0': f'Complications={"无" if complication=="没有" else "有"}',
            'Complications_1': f'Complications={"无" if complication=="没有" else "有"}',
            'Complications_2': f'Complications={"无" if complication=="没有" else "有"}',
            'FTSST': f'FTSST={"慢" if sit_stand=="大于等于12s" else "快"}',
            'Walking_speed': f'Walking speed={"慢" if walk_speed=="小于1m/s" else "快"}',
            'fall': f'Fall={"是" if fall=="是" else "否"}',
            'ADL': f'ADL={"受限" if daily_activity=="有限制" else "正常"}',
            'gender': f'Gender={"女" if gender=="女" else "男"}',
            'PA_high': f'PA={"高" if activity=="高水平" else "中/低"}',
            'PA_medium': f'PA={"中" if activity=="中水平" else "高/低"}',
            'PA_low': f'PA={"低" if activity=="低水平" else "高/中"}',
            'smoking': f'Smoke={"是" if smoking=="是" else "否"}'
        }

        # 创建SHAP决策图
        st.subheader(f"🧠 决策依据分析（{'衰弱' if pred_label == 1 else '非衰弱'}类）")
        plt.close('all')
        fig = plt.figure(figsize=(14, 4))
        
        shap.force_plot(
            base_value=expected_value,
            shap_values=shap_value,
            features=input_df.iloc[0],
            feature_names=[feature_names_mapping.get(f, f) for f in input_df.columns],
            matplotlib=True,
            show=False,
            plot_cmap="RdBu"
        )
        
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
        
    except Exception as e:
        st.error(f"可视化渲染失败，请刷新页面重试。技术细节: {str(e)}")
    
    # 图例说明
    st.markdown("""
    **图例说明:**
    - 🔴 **红色**：增加衰弱风险的特征  
    - 🟢 **绿色**：降低衰弱风险的特征  
    """)

# 页脚
st.markdown("---")
st.caption("©2025 KOA预测系统 | 仅供临床参考")



