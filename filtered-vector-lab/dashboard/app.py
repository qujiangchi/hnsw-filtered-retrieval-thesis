import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 页面配置
st.set_page_config(page_title="实验平台", layout="wide", initial_sidebar_state="expanded")

page = st.query_params.get("page", "overview")

# =============================================================================
# 全局 CSS
# =============================================================================
st.markdown("""
<style>
    /* 基础重置与背景颜色 */
    html, body, [class*="st-"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    .stApp {
        background-color: #f0f2f5 !important;
    }
    header[data-testid="stHeader"] { display: none !important; }
    
    /* ====================================================
       侧边栏重置
       ==================================================== */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e8e8e8 !important;
        min-width: 180px !important;
        max-width: 180px !important;
    }
    div[data-testid="stSidebar"] div[data-testid="stRadio"] {
        display: none !important;
    }
    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 16px !important;
    }
    
    .custom-sidebar {
        padding: 0 12px;
    }
    .sidebar-item {
        padding: 10px 12px;
        margin-bottom: 4px;
        border-radius: 6px;
        color: #595959;
        font-size: 13px;
        cursor: pointer;
        display: block;
        text-decoration: none;
    }
    .sidebar-item:hover {
        background-color: #f5f5f5;
        text-decoration: none;
        color: #595959;
    }
    .sidebar-item.active {
        background-color: #e6f7ff;
        color: #1890ff;
        font-weight: 600;
        position: relative;
    }
    .sidebar-item.active:hover {
        color: #1890ff;
    }
    .sidebar-item.active::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 3px;
        background-color: #1890ff;
    }
    
    /* ====================================================
       主内容区布局
       ==================================================== */
    .main .block-container {
        padding-top: 24px !important;
        padding-left: 32px !important;
        padding-right: 32px !important;
        padding-bottom: 32px !important;
        max-width: 100% !important;
    }
    
    [data-testid="column"] {
        gap: 16px !important;
    }
    [data-testid="stVerticalBlock"] {
        gap: 16px !important;
    }
    
    /* 自定义卡片样式 */
    .custom-card {
        background-color: #ffffff;
        border: 1px solid #d9d9d9;
        border-radius: 8px;
        padding: 16px 20px;
        height: 100%;
    }
    
    /* 图表和内容标题 */
    .card-title {
        font-size: 14px;
        font-weight: 600;
        color: #262626;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .card-title .info-icon {
        color: #bfbfbf;
        font-size: 13px;
        font-weight: normal;
        cursor: help;
    }
    .card-title .subtitle {
        float: right;
        font-size: 12px;
        color: #8c8c8c;
        font-weight: normal;
        margin-left: auto;
    }

    /* Badges */
    .badge-success { color: #52c41a; background: #f6ffed; border: 1px solid #b7eb8f; padding: 2px 8px; border-radius: 4px; font-size: 12px; }
    .badge-warning { color: #faad14; background: #fffbe6; border: 1px solid #ffe58f; padding: 2px 8px; border-radius: 4px; font-size: 12px; }
    .badge-error { color: #ff4d4f; background: #fff2f0; border: 1px solid #ffccc7; padding: 2px 8px; border-radius: 4px; font-size: 12px; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 侧边栏
# =============================================================================
with st.sidebar:
    sidebar_html = f"""
    <div class="custom-sidebar">
        <a href="?page=overview" target="_self" class="sidebar-item {'active' if page == 'overview' else ''}">实验总览</a>
        <a href="?page=dataset" target="_self" class="sidebar-item {'active' if page == 'dataset' else ''}">数据集管理</a>
        <a href="?page=filter_config" target="_self" class="sidebar-item {'active' if page == 'filter_config' else ''}">过滤条件配置</a>
        <a href="?page=build_index" target="_self" class="sidebar-item {'active' if page == 'build_index' else ''}">HNSW 索引构建</a>
        <a href="?page=query_vis" target="_self" class="sidebar-item {'active' if page == 'query_vis' else ''}">查询过程可视化</a>
        <a href="?page=perf_compare" target="_self" class="sidebar-item {'active' if page == 'perf_compare' else ''}">性能对比分析</a>
        <a href="?page=param_sens" target="_self" class="sidebar-item {'active' if page == 'param_sens' else ''}">参数敏感性分析</a>
        <a href="?page=export_report" target="_self" class="sidebar-item {'active' if page == 'export_report' else ''}">实验报告导出</a>
    </div>
    """
    sidebar_html = "".join([line.strip() for line in sidebar_html.split("\n")])
    st.markdown(sidebar_html, unsafe_allow_html=True)


# =============================================================================
# 页面 1: 实验总览 (Overview)
# =============================================================================
def render_overview():
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: flex-end; padding-bottom: 16px; margin-bottom: 16px;">
        <div>
            <div style="font-size: 13px; color: #8c8c8c; margin-bottom: 6px;">首页 / 实验分析 / <span style="color: #262626;">实验总览</span></div>
            <div style="font-size: 22px; font-weight: 600; color: #262626; margin-bottom: 4px;">实验总览</div>
            <div style="font-size: 13px; color: #8c8c8c;">基于 HNSW 的过滤向量检索优化方法研究与实现</div>
        </div>
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="display: flex; align-items: center; gap: 6px; font-size: 13px; color: #595959;">
                <span>数据集</span>
                <select style="border: 1px solid #d9d9d9; border-radius: 4px; padding: 4px 8px; background: #fff; color: #595959; outline: none;"><option>SIFT-1M</option></select>
            </div>
            <div style="display: flex; align-items: center; gap: 6px; font-size: 13px; color: #595959;">
                <span>算法组</span>
                <select style="border: 1px solid #d9d9d9; border-radius: 4px; padding: 4px 8px; background: #fff; color: #595959; outline: none;"><option>全部算法</option></select>
            </div>
            <div style="display: flex; align-items: center; gap: 6px; font-size: 13px; color: #595959;">
                <span>实验轮次</span>
                <select style="border: 1px solid #d9d9d9; border-radius: 4px; padding: 4px 8px; background: #fff; color: #595959; outline: none; width: 140px;"><option>Run-20240520-01</option></select>
            </div>
            <div style="display: flex; align-items: center; gap: 6px; font-size: 13px; color: #595959;">
                <span>批次/日期</span>
                <div style="border: 1px solid #d9d9d9; border-radius: 4px; padding: 4px 8px; background: #fff; display: flex; align-items: center; gap: 4px; color: #595959;">2024-05-20 📅</div>
            </div>
            <div style="display: flex; gap: 8px; margin-left: 8px;">
                <button style="border: 1px solid #d9d9d9; background: #fff; color: #595959; padding: 4px 16px; border-radius: 4px; cursor: pointer; font-size: 13px;">↻ 刷新</button>
                <button style="border: none; background: #1890ff; color: #fff; padding: 4px 16px; border-radius: 4px; cursor: pointer; font-size: 13px;">⬇ 导出</button>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    kpi_html = """
    <div style="background: #fff; border: 1px solid #d9d9d9; border-radius: 8px; display: flex; align-items: center; padding: 20px 0; margin-bottom: 24px;">
        <div style="flex: 1; padding: 0 24px; border-right: 1px solid #f0f0f0;">
            <div style="font-size: 14px; color: #595959; margin-bottom: 8px; display: flex; align-items: center; gap: 4px;">平均 Recall@10 <span style="color:#bfbfbf; font-size: 13px;">ⓘ</span></div>
            <div style="font-size: 26px; font-weight: 500; color: #262626; margin-bottom: 4px;">0.947</div>
            <div style="font-size: 13px;"><span style="color: #1890ff;">↑ 0.012</span> <span style="color: #8c8c8c; margin-left: 4px;">较上次</span></div>
        </div>
        <div style="flex: 1; padding: 0 24px; border-right: 1px solid #f0f0f0;">
            <div style="font-size: 14px; color: #595959; margin-bottom: 8px; display: flex; align-items: center; gap: 4px;">QPS <span style="color:#bfbfbf; font-size: 13px;">ⓘ</span></div>
            <div style="font-size: 26px; font-weight: 500; color: #262626; margin-bottom: 4px;">1284</div>
            <div style="font-size: 13px;"><span style="color: #1890ff;">↑ 8.7%</span> <span style="color: #8c8c8c; margin-left: 4px;">较上次</span></div>
        </div>
        <div style="flex: 1; padding: 0 24px; border-right: 1px solid #f0f0f0;">
            <div style="font-size: 14px; color: #595959; margin-bottom: 8px; display: flex; align-items: center; gap: 4px;">P95 延迟(ms) <span style="color:#bfbfbf; font-size: 13px;">ⓘ</span></div>
            <div style="font-size: 26px; font-weight: 500; color: #262626; margin-bottom: 4px;">23.8</div>
            <div style="font-size: 13px;"><span style="color: #1890ff;">↓ 9.1%</span> <span style="color: #8c8c8c; margin-left: 4px;">较上次</span></div>
        </div>
        <div style="flex: 1; padding: 0 24px; border-right: 1px solid #f0f0f0;">
            <div style="font-size: 14px; color: #595959; margin-bottom: 8px; display: flex; align-items: center; gap: 4px;">内存占用(GB) <span style="color:#bfbfbf; font-size: 13px;">ⓘ</span></div>
            <div style="font-size: 26px; font-weight: 500; color: #262626; margin-bottom: 4px;">6.42</div>
            <div style="font-size: 13px;"><span style="color: #1890ff;">↑ 3.2%</span> <span style="color: #8c8c8c; margin-left: 4px;">较上次</span></div>
        </div>
        <div style="flex: 1; padding: 0 24px; border-right: 1px solid #f0f0f0;">
            <div style="font-size: 14px; color: #595959; margin-bottom: 8px; display: flex; align-items: center; gap: 4px;">索引构建时间(min) <span style="color:#bfbfbf; font-size: 13px;">ⓘ</span></div>
            <div style="font-size: 26px; font-weight: 500; color: #262626; margin-bottom: 4px;">18.6</div>
            <div style="font-size: 13px;"><span style="color: #1890ff;">↓ 6.4%</span> <span style="color: #8c8c8c; margin-left: 4px;">较上次</span></div>
        </div>
        <div style="flex: 1; padding: 0 24px;">
            <div style="font-size: 14px; color: #595959; margin-bottom: 8px; display: flex; align-items: center; gap: 4px;">过滤选择率 <span style="color:#bfbfbf; font-size: 13px;">ⓘ</span></div>
            <div style="font-size: 26px; font-weight: 500; color: #262626; margin-bottom: 4px;">12.5%</div>
            <div style="font-size: 13px;"><span style="color: #1890ff;">↓ 2.1%</span> <span style="color: #8c8c8c; margin-left: 4px;">较上次</span></div>
        </div>
    </div>
    """
    kpi_html = "".join([line.strip() for line in kpi_html.split("\n")])
    st.markdown(kpi_html, unsafe_allow_html=True)

    col1, col2 = st.columns([2.2, 1])

    with col1:
        st.markdown('<div class="custom-card"><div class="card-title">Recall-QPS 性能曲线 <span class="info-icon">ⓘ</span><div class="subtitle">数据集: SIFT-1M / Top-k=10</div></div>', unsafe_allow_html=True)
        
        fig1 = go.Figure()
        recalls = np.linspace(0.7, 1.0, 15)
        y_hnsw = 10000 * np.exp(-10 * (recalls - 0.7))
        fig1.add_trace(go.Scatter(x=recalls, y=y_hnsw, name="HNSW-Base", mode="lines+markers", line=dict(color="#1f4499", width=2), marker=dict(symbol="circle", size=6)))
        fig1.add_trace(go.Scatter(x=recalls, y=y_hnsw * 0.4, name="Post-Filter", mode="lines+markers", line=dict(color="#69b1ff", width=2, dash="dash"), marker=dict(symbol="circle-open", size=6)))
        fig1.add_trace(go.Scatter(x=recalls, y=y_hnsw * 1.5, name="Window Search Tree", mode="lines+markers", line=dict(color="#40a9ff", width=2, dash="dashdot"), marker=dict(symbol="triangle-up", size=6)))
        fig1.add_trace(go.Scatter(x=recalls, y=y_hnsw * 2.2, name="SIEVE", mode="lines+markers", line=dict(color="#91d5ff", width=2), marker=dict(symbol="square", size=6)))
        fig1.add_trace(go.Scatter(x=recalls, y=y_hnsw * 3.5, name="Optimized HNSW", mode="lines+markers", line=dict(color="#001529", width=2), marker=dict(symbol="circle", size=6)))
        
        fig1.update_layout(
            height=300, margin=dict(l=0, r=0, t=10, b=0),
            yaxis_type="log", plot_bgcolor="#fff", paper_bgcolor="#fff",
            legend=dict(orientation="v", yanchor="bottom", y=0.05, xanchor="right", x=0.98, bgcolor="rgba(255,255,255,0.8)", bordercolor="#e8e8e8", borderwidth=1, font=dict(size=11)),
            xaxis=dict(range=[0.68, 1.02], showgrid=True, gridcolor="#f0f0f0", tickformat=".2f", title="Recall@10", title_font=dict(size=12, color="#595959")),
            yaxis=dict(range=[1, 5.2], showgrid=True, gridcolor="#f0f0f0", tickformat="d", title="QPS (次/秒)", title_font=dict(size=12, color="#595959")),
        )
        
        fig1.add_annotation(
            x=0.90, y=np.log10(60000), text="<b>R=0.90</b><br>HNSW-Base: 1420<br>Opt. HNSW: 2360<br>提升: <b>66.2%</b>",
            showarrow=False, align="left", xanchor="left", yanchor="top", bgcolor="#fff", bordercolor="#1890ff", borderwidth=1, borderpad=8, font=dict(size=11, color="#262626")
        )
        fig1.add_annotation(
            x=0.95, y=np.log10(60000), text="<b>R=0.95</b><br>HNSW-Base: 780<br>Opt. HNSW: 1284<br>提升: <b>64.6%</b>",
            showarrow=False, align="left", xanchor="left", yanchor="top", bgcolor="#fff", bordercolor="#1890ff", borderwidth=1, borderpad=8, font=dict(size=11, color="#262626")
        )
        
        st.plotly_chart(fig1, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="custom-card"><div class="card-title">当前实验配置</div>', unsafe_allow_html=True)
        config_html = """
        <table style="width: 100%; font-size: 13px; border-collapse: collapse;">
            <tr><td style="padding: 6.5px 0; color: #595959; border-bottom: 1px solid #f0f0f0; width: 45%;">数据集</td><td style="padding: 6.5px 0; color: #262626; border-bottom: 1px solid #f0f0f0;">SIFT-1M</td></tr>
            <tr><td style="padding: 6.5px 0; color: #595959; border-bottom: 1px solid #f0f0f0;">数据集规模</td><td style="padding: 6.5px 0; color: #262626; border-bottom: 1px solid #f0f0f0;">1,000,000 向量</td></tr>
            <tr><td style="padding: 6.5px 0; color: #595959; border-bottom: 1px solid #f0f0f0;">向量维度</td><td style="padding: 6.5px 0; color: #262626; border-bottom: 1px solid #f0f0f0;">128</td></tr>
            <tr><td style="padding: 6.5px 0; color: #595959; border-bottom: 1px solid #f0f0f0;">查询数量</td><td style="padding: 6.5px 0; color: #262626; border-bottom: 1px solid #f0f0f0;">10,000</td></tr>
            <tr><td style="padding: 6.5px 0; color: #595959; border-bottom: 1px solid #f0f0f0;">过滤字段</td><td style="padding: 6.5px 0; color: #262626; border-bottom: 1px solid #f0f0f0;">category (基数=100)</td></tr>
            <tr><td style="padding: 6.5px 0; color: #595959; border-bottom: 1px solid #f0f0f0;">Top-k</td><td style="padding: 6.5px 0; color: #262626; border-bottom: 1px solid #f0f0f0;">10</td></tr>
            <tr><td style="padding: 6.5px 0; color: #595959; border-bottom: 1px solid #f0f0f0;">M</td><td style="padding: 6.5px 0; color: #262626; border-bottom: 1px solid #f0f0f0;">16</td></tr>
            <tr><td style="padding: 6.5px 0; color: #595959; border-bottom: 1px solid #f0f0f0;">efConstruction</td><td style="padding: 6.5px 0; color: #262626; border-bottom: 1px solid #f0f0f0;">200</td></tr>
            <tr><td style="padding: 6.5px 0; color: #595959; border-bottom: 1px solid #f0f0f0;">efSearch</td><td style="padding: 6.5px 0; color: #262626; border-bottom: 1px solid #f0f0f0;">100</td></tr>
            <tr><td style="padding: 6.5px 0; color: #595959; border-bottom: 1px solid #f0f0f0;">当前轮次 ID</td><td style="padding: 6.5px 0; color: #262626; border-bottom: 1px solid #f0f0f0;">Run-20240520-01</td></tr>
            <tr><td style="padding: 6.5px 0; color: #595959;">硬件节点</td><td style="padding: 6.5px 0; color: #262626;">Intel Xeon Gold 6338 x 2, 256GB RAM</td></tr>
        </table>
        </div>
        """
        config_html = "".join([line.strip() for line in config_html.split("\n")])
        st.markdown(config_html, unsafe_allow_html=True)

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    col3, col4, col5 = st.columns([1, 1, 1])

    with col3:
        st.markdown('<div class="custom-card"><div class="card-title">不同过滤选择率下延迟变化 <span class="info-icon">ⓘ</span></div>', unsafe_allow_html=True)
        
        fig2 = go.Figure()
        selectivity = ["1%", "5%", "10%", "25%", "50%", "100%"]
        
        fig2.add_trace(go.Scatter(x=selectivity, y=[50, 60, 80, 100, 150, 200], name="HNSW-Base", mode="lines+markers", line=dict(color="#1f4499", width=2), marker=dict(symbol="circle", size=6)))
        fig2.add_trace(go.Scatter(x=selectivity, y=[80, 100, 130, 180, 250, 300], name="Post-Filter", mode="lines+markers", line=dict(color="#69b1ff", width=2, dash="dash"), marker=dict(symbol="circle-open", size=6)))
        fig2.add_trace(go.Scatter(x=selectivity, y=[15, 25, 40, 70, 120, 180], name="Window Search Tree", mode="lines+markers", line=dict(color="#40a9ff", width=2, dash="dashdot"), marker=dict(symbol="triangle-up", size=6)))
        fig2.add_trace(go.Scatter(x=selectivity, y=[10, 18, 30, 50, 80, 120], name="SIEVE", mode="lines+markers", line=dict(color="#91d5ff", width=2), marker=dict(symbol="square", size=6)))
        fig2.add_trace(go.Scatter(x=selectivity, y=[5, 10, 18, 30, 50, 80], name="Optimized HNSW", mode="lines+markers", line=dict(color="#001529", width=2), marker=dict(symbol="circle", size=6)))
        
        fig2.update_layout(
            height=260, margin=dict(l=0, r=0, t=10, b=0),
            yaxis_type="log", plot_bgcolor="#fff", paper_bgcolor="#fff",
            legend=dict(orientation="v", yanchor="top", y=0.9, xanchor="right", x=0.95, font=dict(size=10), bgcolor="rgba(255,255,255,0)"),
            xaxis=dict(showgrid=True, gridcolor="#f0f0f0", title="过滤选择率 (%)", title_font=dict(size=12, color="#595959")),
            yaxis=dict(showgrid=True, gridcolor="#f0f0f0", title="P95 延迟 (ms)", title_font=dict(size=12, color="#595959")),
        )
        st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="custom-card"><div class="card-title">参数敏感性分析 <span class="info-icon">ⓘ</span><div class="subtitle">数据集: SIFT-1M / 选择率=10%</div></div>', unsafe_allow_html=True)
        
        fig3 = make_subplots(specs=[[{"secondary_y": True}]])
        x_params = ["8<br>20", "16<br>50", "24<br>100", "32<br>200", "48<br>400", "64<br>800"]
        
        fig3.add_trace(go.Scatter(x=x_params, y=[0.82, 0.91, 0.96, 0.98, 0.99, 1.0], name="Recall@10 (左轴)", mode="lines+markers", line=dict(color="#1890ff", width=2), marker=dict(symbol="circle", size=6)), secondary_y=False)
        fig3.add_trace(go.Scatter(x=x_params, y=[10, 15, 22, 35, 45, 58], name="P95 延迟 (右轴)", mode="lines+markers", line=dict(color="#001529", width=2, dash="dash"), marker=dict(symbol="square-open", size=6)), secondary_y=True)
        
        fig3.update_layout(
            height=260, margin=dict(l=0, r=0, t=30, b=20),
            plot_bgcolor="#fff", paper_bgcolor="#fff",
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5, font=dict(size=10)),
            xaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
        )
        fig3.update_yaxes(title_text="Recall@10", range=[0.8, 1.0], showgrid=True, gridcolor="#f0f0f0", secondary_y=False, title_font=dict(size=12, color="#595959"))
        fig3.update_yaxes(title_text="P95 延迟(ms)", range=[0, 60], showgrid=False, secondary_y=True, title_font=dict(size=12, color="#595959"))
        
        fig3.add_annotation(x=-0.1, y=-0.15, text="M", showarrow=False, xref="paper", yref="paper", font=dict(size=11, color="#595959"))
        fig3.add_annotation(x=-0.1, y=-0.25, text="efSearch", showarrow=False, xref="paper", yref="paper", font=dict(size=11, color="#595959"))
        
        st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col5:
        st.markdown('<div class="custom-card"><div class="card-title">索引资源占用 <span class="info-icon">ⓘ</span><div class="subtitle">数据集: SIFT-1M / 选择率=10%</div></div>', unsafe_allow_html=True)
        
        fig4 = make_subplots(specs=[[{"secondary_y": True}]])
        algos = ["HNSW-Base", "Post-Filter", "Window<br>Search Tree", "SIEVE", "Optimized<br>HNSW"]
        
        fig4.add_trace(go.Bar(x=algos, y=[5.88, 6.15, 7.26, 6.98, 6.42], name="内存占用 (GB)", marker_color="#1f4499", text=[5.88, 6.15, 7.26, 6.98, 6.42], textposition="outside", textfont=dict(size=10)), secondary_y=False)
        fig4.add_trace(go.Bar(x=algos, y=[17.2, 17.9, 21.3, 19.5, 18.6], name="构建时间 (min)", marker_color="#91d5ff", text=[17.2, 17.9, 21.3, 19.5, 18.6], textposition="outside", textfont=dict(size=10)), secondary_y=True)
        
        fig4.update_layout(
            height=260, margin=dict(l=0, r=0, t=30, b=0),
            plot_bgcolor="#fff", paper_bgcolor="#fff",
            barmode='group', bargap=0.3,
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5, font=dict(size=10))
        )
        fig4.update_yaxes(title_text="内存占用 (GB)", range=[0, 10], showgrid=True, gridcolor="#f0f0f0", secondary_y=False, title_font=dict(size=12, color="#595959"))
        fig4.update_yaxes(title_text="构建时间 (min)", range=[0, 50], showgrid=False, secondary_y=True, title_font=dict(size=12, color="#595959"))
        
        st.plotly_chart(fig4, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    col6, col7 = st.columns([1.1, 2])

    with col6:
        st.markdown('<div class="custom-card"><div class="card-title">HNSW 查询过程概览 <span class="info-icon">ⓘ</span></div>', unsafe_allow_html=True)
        hnsw_html = """
        <div style="width: 100%; height: 260px; display: flex; flex-direction: column; justify-content: center; position: relative; font-family: sans-serif;">
            <div style="display: flex; gap: 16px; margin-bottom: 30px; font-size: 12px; color: #595959; justify-content: center;">
                <div style="display: flex; align-items: center; gap: 4px;"><div style="width:12px; height:12px; border-radius:50%; background:#e8e8e8; border: 1px solid #bfbfbf;"></div> 入口点</div>
                <div style="display: flex; align-items: center; gap: 4px;"><div style="width:12px; height:12px; border-radius:50%; background:#e6f7ff; border: 1px solid #1890ff;"></div> 已访问节点</div>
                <div style="display: flex; align-items: center; gap: 4px;"><div style="width:12px; height:12px; border-radius:50%; background:#fff; border: 1px dashed #1890ff;"></div> 候选队列</div>
                <div style="display: flex; align-items: center; gap: 4px;"><div style="width:12px; height:12px; border-radius:50%; background:#f6ffed; border: 1px solid #52c41a;"></div> 最终 Top-k</div>
                <div style="display: flex; align-items: center; gap: 4px;"> <span style="color:#1890ff;">→</span> 查询路径</div>
            </div>
            <div style="display: flex; justify-content: space-around; align-items: center; padding: 0 16px;">
                <div style="position: relative; width: 120px; height: 120px;">
                    <div style="position: absolute; top: 10px; left: 50px; width:16px; height:16px; border-radius:50%; background:#e8e8e8; border: 1px solid #bfbfbf;"></div>
                    <div style="position: absolute; top: 40px; left: 20px; width:16px; height:16px; border-radius:50%; background:#e6f7ff; border: 1px solid #1890ff;"></div>
                    <div style="position: absolute; top: 40px; left: 80px; width:16px; height:16px; border-radius:50%; background:#e6f7ff; border: 1px solid #1890ff;"></div>
                    <div style="position: absolute; top: 80px; left: 30px; width:16px; height:16px; border-radius:50%; background:#e8e8e8; border: 1px solid #bfbfbf;"></div>
                    <div style="position: absolute; top: 80px; left: 70px; width:16px; height:16px; border-radius:50%; background:#e6f7ff; border: 1px solid #1890ff;"></div>
                    <div style="position: absolute; top: 100px; left: 100px; width:16px; height:16px; border-radius:50%; background:#e8e8e8; border: 1px solid #bfbfbf;"></div>
                    <svg width="120" height="120" style="position:absolute; top:0; left:0; z-index:-1;">
                        <line x1="58" y1="18" x2="28" y2="48" stroke="#1890ff" stroke-width="1.5" />
                        <line x1="58" y1="18" x2="88" y2="48" stroke="#1890ff" stroke-width="1.5" />
                        <line x1="28" y1="48" x2="38" y2="88" stroke="#bfbfbf" stroke-width="1" />
                        <line x1="28" y1="48" x2="78" y2="88" stroke="#1890ff" stroke-width="1.5" />
                        <line x1="88" y1="48" x2="78" y2="88" stroke="#1890ff" stroke-width="1.5" />
                        <line x1="88" y1="48" x2="108" y2="108" stroke="#bfbfbf" stroke-width="1" />
                    </svg>
                    <div style="position:absolute; top:-20px; left: 20px; font-size:11px; color:#8c8c8c;">入口点</div>
                    <div style="position:absolute; top:-5px; left: 45px; font-size:14px; color:#1890ff;">↓</div>
                </div>
                <div style="display: flex; flex-direction: column; align-items: center; gap: 8px; border: 1px dashed #bfbfbf; padding: 16px; border-radius: 4px; position: relative;">
                    <div style="position:absolute; top:-25px; font-size:12px; color:#595959; white-space:nowrap; font-weight: 500;">候选队列</div>
                    <div style="width:16px; height:16px; border-radius:50%; background:#fff; border: 1px dashed #1890ff;"></div>
                    <div style="width:16px; height:16px; border-radius:50%; background:#fff; border: 1px dashed #1890ff;"></div>
                    <div style="color: #bfbfbf; font-size: 14px; line-height: 1; margin: 4px 0;">...</div>
                    <div style="width:16px; height:16px; border-radius:50%; background:#fff; border: 1px dashed #1890ff;"></div>
                </div>
                <div style="display: flex; flex-direction: column; align-items: center; gap: 8px; border: 1px solid #e8e8e8; padding: 16px; border-radius: 4px; background: #fafafa; position: relative;">
                    <div style="position:absolute; top:-25px; font-size:12px; color:#595959; white-space:nowrap; font-weight: 500;">Top-k 结果</div>
                    <div style="width:16px; height:16px; border-radius:50%; background:#f6ffed; border: 1px solid #52c41a;"></div>
                    <div style="width:16px; height:16px; border-radius:50%; background:#f6ffed; border: 1px solid #52c41a;"></div>
                    <div style="width:16px; height:16px; border-radius:50%; background:#f6ffed; border: 1px solid #52c41a;"></div>
                </div>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 0 40px; margin-top: 30px; font-size: 12px; color: #8c8c8c;">
                <div>入口层搜索</div>
                <div>邻居扩展</div>
                <div>候选筛选</div>
                <div>结果返回</div>
            </div>
        </div>
        </div>
        """
        hnsw_html = "".join([line.strip() for line in hnsw_html.split("\n")])
        st.markdown(hnsw_html, unsafe_allow_html=True)

    with col7:
        st.markdown('<div class="custom-card"><div class="card-title">实验明细表 <span class="info-icon">ⓘ</span><button style="float:right; border: 1px solid #d9d9d9; background: #fff; padding: 2px 12px; border-radius: 4px; font-size: 12px; cursor: pointer; color: #595959; font-weight: normal; margin-left: auto;">查看全部</button></div>', unsafe_allow_html=True)
        table_html = """
        <table style="width: 100%; border-collapse: collapse; font-size: 13px; text-align: center; margin-bottom: 16px;">
            <tr style="background: #fafafa; color: #595959;">
                <th style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; font-weight: 500;">轮次</th>
                <th style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; font-weight: 500;">方法</th>
                <th style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; font-weight: 500;">数据集</th>
                <th style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; font-weight: 500;">选择率</th>
                <th style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; font-weight: 500;">Recall@10</th>
                <th style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; font-weight: 500;">QPS (次/秒)</th>
                <th style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; font-weight: 500;">P95延迟 (ms)</th>
                <th style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; font-weight: 500;">内存 (GB)</th>
                <th style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; font-weight: 500;">构建时间 (min)</th>
                <th style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; font-weight: 500;">状态</th>
            </tr>
            <tr>
                <td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; color: #595959;">Run-20240520-01</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; color: #262626;">Optimized HNSW</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; color: #595959;">SIFT-1M</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; color: #595959;">10%</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; color: #262626;">0.957</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; color: #262626;">1284</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; color: #262626;">23.8</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; color: #262626;">6.42</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; color: #262626;">18.6</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0;"><span class="badge-success">完成</span></td>
            </tr>
            <tr>
                <td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; color: #595959;">Run-20240520-01</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; color: #262626;">HNSW-Base</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; color: #595959;">SIFT-1M</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; color: #595959;">10%</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; color: #262626;">0.951</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; color: #262626;">780</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; color: #262626;">32.6</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; color: #262626;">5.88</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; color: #262626;">17.2</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0;"><span class="badge-success">完成</span></td>
            </tr>
            <tr>
                <td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; color: #595959;">Run-20240520-01</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; color: #262626;">Post-Filter</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; color: #595959;">SIFT-1M</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; color: #595959;">10%</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; color: #262626;">0.918</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; color: #262626;">312</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; color: #262626;">68.9</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; color: #262626;">6.15</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; color: #262626;">17.9</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0;"><span class="badge-success">完成</span></td>
            </tr>
            <tr>
                <td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; color: #595959;">Run-20240520-01</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; color: #262626;">Window Search Tree</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; color: #595959;">SIFT-1M</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; color: #595959;">10%</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; color: #262626;">0.933</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; color: #262626;">456</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; color: #262626;">45.7</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; color: #262626;">7.26</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; color: #262626;">21.3</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0;"><span class="badge-success">完成</span></td>
            </tr>
            <tr>
                <td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; color: #595959;">Run-20240520-01</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; color: #262626;">SIEVE</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; color: #595959;">SIFT-1M</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; color: #595959;">10%</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; color: #262626;">0.940</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; color: #262626;">602</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; color: #262626;">35.4</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; color: #262626;">6.98</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; color: #262626;">19.5</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0;"><span class="badge-success">完成</span></td>
            </tr>
            <tr>
                <td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; color: #595959;">Run-20240519-02</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; color: #262626;">Optimized HNSW</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; color: #595959;">SIFT-1M</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; color: #595959;">5%</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; color: #262626;">0.961</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; color: #262626;">1460</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; color: #262626;">21.6</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; color: #262626;">6.38</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; color: #262626;">18.4</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0;"><span class="badge-success">完成</span></td>
            </tr>
        </table>
        
        <div style="display: flex; justify-content: flex-end; align-items: center; font-size: 13px; color: #595959; gap: 16px;">
            <div>共 24 条</div>
            <div style="display: flex; gap: 4px;">
                <div style="border: 1px solid #d9d9d9; padding: 2px 8px; border-radius: 4px; cursor: pointer; color: #bfbfbf;">&lt;</div>
                <div style="border: 1px solid #1890ff; background: #fff; color: #1890ff; padding: 2px 8px; border-radius: 4px; cursor: pointer;">1</div>
                <div style="border: 1px solid #d9d9d9; background: #fff; padding: 2px 8px; border-radius: 4px; cursor: pointer;">2</div>
                <div style="border: 1px solid #d9d9d9; background: #fff; padding: 2px 8px; border-radius: 4px; cursor: pointer;">3</div>
                <div style="border: 1px solid #d9d9d9; background: #fff; padding: 2px 8px; border-radius: 4px; cursor: pointer;">4</div>
                <div style="border: 1px solid #d9d9d9; padding: 2px 8px; border-radius: 4px; cursor: pointer;">&gt;</div>
            </div>
            <select style="border: 1px solid #d9d9d9; border-radius: 4px; padding: 2px 4px; background: #fff; outline: none; color: #595959;">
                <option>10 条/页</option>
            </select>
        </div>
        </div>
        """
        table_html = "".join([line.strip() for line in table_html.split("\n")])
        st.markdown(table_html, unsafe_allow_html=True)


# =============================================================================
# 页面 2: 数据集管理 (Dataset Management)
# =============================================================================
def render_dataset():
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: flex-end; padding-bottom: 16px; margin-bottom: 16px;">
        <div>
            <div style="font-size: 13px; color: #8c8c8c; margin-bottom: 6px;">首页 / 实验分析 / <span style="color: #262626;">数据集管理</span></div>
            <div style="font-size: 22px; font-weight: 600; color: #262626; margin-bottom: 4px;">数据集管理</div>
            <div style="font-size: 13px; color: #8c8c8c;">基于 HNSW 的过滤向量检索优化方法研究与实现</div>
        </div>
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="display: flex; align-items: center; gap: 6px; font-size: 13px; color: #595959;">
                <span>数据集</span>
                <select style="border: 1px solid #d9d9d9; border-radius: 4px; padding: 4px 8px; background: #fff; color: #595959; outline: none;"><option>SIFT-1M</option></select>
            </div>
            <div style="display: flex; align-items: center; gap: 6px; font-size: 13px; color: #595959;">
                <span>数据源</span>
                <select style="border: 1px solid #d9d9d9; border-radius: 4px; padding: 4px 8px; background: #fff; color: #595959; outline: none;"><option>ANN-Benchmarks</option></select>
            </div>
            <div style="display: flex; align-items: center; gap: 6px; font-size: 13px; color: #595959;">
                <span>算法输出</span>
                <select style="border: 1px solid #d9d9d9; border-radius: 4px; padding: 4px 8px; background: #fff; color: #595959; outline: none;"><option>全部项目</option></select>
            </div>
            <div style="display: flex; align-items: center; gap: 6px; font-size: 13px; color: #595959;">
                <span>批次</span>
                <select style="border: 1px solid #d9d9d9; border-radius: 4px; padding: 4px 8px; background: #fff; color: #595959; outline: none;"><option>2024-05-20</option></select>
            </div>
            <div style="display: flex; gap: 8px; margin-left: 8px;">
                <button style="border: 1px solid #d9d9d9; background: #fff; color: #595959; padding: 4px 16px; border-radius: 4px; cursor: pointer; font-size: 13px;">↻ 刷新</button>
                <button style="border: none; background: #1890ff; color: #fff; padding: 4px 16px; border-radius: 4px; cursor: pointer; font-size: 13px;">导入数据集</button>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    kpi_html = """
    <div style="background: #fff; border: 1px solid #d9d9d9; border-radius: 8px; display: flex; align-items: center; padding: 20px 0; margin-bottom: 24px;">
        <div style="flex: 1; padding: 0 24px; border-right: 1px solid #f0f0f0;">
            <div style="font-size: 14px; color: #595959; margin-bottom: 8px; display: flex; align-items: center; gap: 4px;">数据集总数 <span style="color:#bfbfbf; font-size: 13px;">ⓘ</span></div>
            <div style="font-size: 26px; font-weight: 500; color: #262626; margin-bottom: 4px;">5</div>
            <div style="font-size: 13px;"><span style="color: #1890ff;">↑ 1 个</span> <span style="color: #8c8c8c; margin-left: 4px;">较上次</span></div>
        </div>
        <div style="flex: 1; padding: 0 24px; border-right: 1px solid #f0f0f0;">
            <div style="font-size: 14px; color: #595959; margin-bottom: 8px; display: flex; align-items: center; gap: 4px;">向量总量 <span style="color:#bfbfbf; font-size: 13px;">ⓘ</span></div>
            <div style="font-size: 26px; font-weight: 500; color: #262626; margin-bottom: 4px;">24.68M</div>
            <div style="font-size: 13px;"><span style="color: #1890ff;">↑ 12.4%</span> <span style="color: #8c8c8c; margin-left: 4px;">较上次</span></div>
        </div>
        <div style="flex: 1; padding: 0 24px; border-right: 1px solid #f0f0f0;">
            <div style="font-size: 14px; color: #595959; margin-bottom: 8px; display: flex; align-items: center; gap: 4px;">平均维度 <span style="color:#bfbfbf; font-size: 13px;">ⓘ</span></div>
            <div style="font-size: 26px; font-weight: 500; color: #262626; margin-bottom: 4px;">187</div>
            <div style="font-size: 13px;"><span style="color: #1890ff;">SIFT 128 / CLIP 512</span></div>
        </div>
        <div style="flex: 1; padding: 0 24px; border-right: 1px solid #f0f0f0;">
            <div style="font-size: 14px; color: #595959; margin-bottom: 8px; display: flex; align-items: center; gap: 4px;">标签字段数 <span style="color:#bfbfbf; font-size: 13px;">ⓘ</span></div>
            <div style="font-size: 26px; font-weight: 500; color: #262626; margin-bottom: 4px;">9</div>
            <div style="font-size: 13px;"><span style="color: #1890ff;">category / time / price</span></div>
        </div>
        <div style="flex: 1; padding: 0 24px; border-right: 1px solid #f0f0f0;">
            <div style="font-size: 14px; color: #595959; margin-bottom: 8px; display: flex; align-items: center; gap: 4px;">已完成转换 <span style="color:#bfbfbf; font-size: 13px;">ⓘ</span></div>
            <div style="font-size: 26px; font-weight: 500; color: #262626; margin-bottom: 4px;">92.6%</div>
            <div style="font-size: 13px;"><span style="color: #1890ff;">↑ 5.8%</span> <span style="color: #8c8c8c; margin-left: 4px;">较上次</span></div>
        </div>
        <div style="flex: 1; padding: 0 24px;">
            <div style="font-size: 14px; color: #595959; margin-bottom: 8px; display: flex; align-items: center; gap: 4px;">异常记录率 <span style="color:#bfbfbf; font-size: 13px;">ⓘ</span></div>
            <div style="font-size: 26px; font-weight: 500; color: #262626; margin-bottom: 4px;">0.18%</div>
            <div style="font-size: 13px;"><span style="color: #1890ff;">↓ 0.04%</span> <span style="color: #8c8c8c; margin-left: 4px;">较上次</span></div>
        </div>
    </div>
    """
    kpi_html = "".join([line.strip() for line in kpi_html.split("\n")])
    st.markdown(kpi_html, unsafe_allow_html=True)

    st.markdown('<div class="custom-card" style="padding-bottom: 24px;">'
                '<div class="card-title" style="margin-bottom: 8px;">数据集注册表 <span class="info-icon">ⓘ</span></div>'
                '<div style="font-size: 13px; color: #8c8c8c; margin-bottom: 20px;">来源包含 ANN-Benchmarks、RangeFilteredANN、SIEVE 与本地构造数据，字段统一映射到 vector + scalar attributes。</div>', unsafe_allow_html=True)
    
    table_html = """
    <table style="width: 100%; border-collapse: collapse; font-size: 14px; text-align: left; margin-bottom: 16px;">
        <tr style="color: #8c8c8c; border-bottom: 1px solid #f0f0f0;">
            <th style="padding: 12px 16px; font-weight: 500;">数据集</th>
            <th style="padding: 12px 16px; font-weight: 500;">来源项目</th>
            <th style="padding: 12px 16px; font-weight: 500;">规模</th>
            <th style="padding: 12px 16px; font-weight: 500;">维度</th>
            <th style="padding: 12px 16px; font-weight: 500;">标签字段</th>
            <th style="padding: 12px 16px; font-weight: 500;">状态</th>
        </tr>
        <tr style="border-bottom: 1px solid #f0f0f0;">
            <td style="padding: 16px; color: #262626; font-weight: 500;">SIFT-1M</td>
            <td style="padding: 16px; color: #595959;">ANN-Benchmarks</td>
            <td style="padding: 16px; color: #595959;">1.00M</td>
            <td style="padding: 16px; color: #595959;">128</td>
            <td style="padding: 16px; color: #595959;">category</td>
            <td style="padding: 16px;"><span class="badge-success">就绪</span></td>
        </tr>
        <tr style="border-bottom: 1px solid #f0f0f0; background-color: #fafafa;">
            <td style="padding: 16px; color: #262626; font-weight: 500;">GloVe-1.18M</td>
            <td style="padding: 16px; color: #595959;">ANN-Benchmarks</td>
            <td style="padding: 16px; color: #595959;">1.18M</td>
            <td style="padding: 16px; color: #595959;">100</td>
            <td style="padding: 16px; color: #595959;">source / lang</td>
            <td style="padding: 16px;"><span class="badge-success">就绪</span></td>
        </tr>
        <tr style="border-bottom: 1px solid #f0f0f0;">
            <td style="padding: 16px; color: #262626; font-weight: 500;">Deep-10M</td>
            <td style="padding: 16px; color: #595959;">ANN-Benchmarks</td>
            <td style="padding: 16px; color: #595959;">9.90M</td>
            <td style="padding: 16px; color: #595959;">96</td>
            <td style="padding: 16px; color: #595959;">bucket</td>
            <td style="padding: 16px;"><span class="badge-warning">转换中</span></td>
        </tr>
        <tr style="border-bottom: 1px solid #f0f0f0; background-color: #fafafa;">
            <td style="padding: 16px; color: #262626; font-weight: 500;">Redcaps-11.6M</td>
            <td style="padding: 16px; color: #595959;">RangeFilteredANN</td>
            <td style="padding: 16px; color: #595959;">11.6M</td>
            <td style="padding: 16px; color: #595959;">512</td>
            <td style="padding: 16px; color: #595959;">date / category</td>
            <td style="padding: 16px;"><span class="badge-success">就绪</span></td>
        </tr>
        <tr style="border-bottom: 1px solid #f0f0f0;">
            <td style="padding: 16px; color: #262626; font-weight: 500;">YFCC-100M</td>
            <td style="padding: 16px; color: #595959;">SIEVE</td>
            <td style="padding: 16px; color: #595959;">100.0M</td>
            <td style="padding: 16px; color: #595959;">128</td>
            <td style="padding: 16px; color: #595959;">camera / year</td>
            <td style="padding: 16px;"><span class="badge-error">失败</span></td>
        </tr>
    </table>
    
    <div style="display: flex; justify-content: flex-end; align-items: center; font-size: 13px; color: #595959; gap: 16px; padding-right: 16px;">
        <div>共 5 条</div>
        <div style="display: flex; gap: 4px;">
            <div style="border: 1px solid #d9d9d9; padding: 2px 8px; border-radius: 4px; cursor: pointer; color: #bfbfbf;">&lt;</div>
            <div style="border: 1px solid #1890ff; background: #fff; color: #1890ff; padding: 2px 8px; border-radius: 4px; cursor: pointer;">1</div>
            <div style="border: 1px solid #d9d9d9; padding: 2px 8px; border-radius: 4px; cursor: pointer;">&gt;</div>
        </div>
        <select style="border: 1px solid #d9d9d9; border-radius: 4px; padding: 2px 4px; background: #fff; outline: none; color: #595959;">
            <option>10 条/页</option>
        </select>
    </div>
    """
    table_html = "".join([line.strip() for line in table_html.split("\n")])
    st.markdown(table_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# 根据路由渲染不同页面
if page == "dataset":
    render_dataset()
elif page == "build_index":
    render_build_index()
elif page == "query_vis":
    render_query_vis()
elif page == "perf_compare":
    render_perf_compare()
elif page == "param_sens":
    render_param_sens()
elif page == "export_report":
    render_export_report()
elif page == "filter_config":
    st.markdown("<h3>过滤条件配置 (开发中...)</h3>", unsafe_allow_html=True)
else:
    render_overview()

# =============================================================================
# 页面 3: 查询过程可视化 (Query Visualization)
# =============================================================================
def render_query_vis():
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: flex-end; padding-bottom: 16px; margin-bottom: 16px;">
        <div>
            <div style="font-size: 13px; color: #8c8c8c; margin-bottom: 6px;">首页 / 实验分析 / <span style="color: #262626;">查询过程可视化</span></div>
            <div style="font-size: 22px; font-weight: 600; color: #262626; margin-bottom: 4px;">查询过程可视化</div>
            <div style="font-size: 13px; color: #8c8c8c;">细粒度展示过滤向量检索的单查询路径，候选队列、访问节点和 Top-k 结果</div>
        </div>
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="display: flex; align-items: center; gap: 6px; font-size: 13px; color: #595959;">
                <span>数据集</span>
                <select style="border: 1px solid #d9d9d9; border-radius: 4px; padding: 4px 8px; background: #fff; color: #595959; outline: none;"><option>Deep1B</option></select>
            </div>
            <div style="display: flex; align-items: center; gap: 6px; font-size: 13px; color: #595959;">
                <span>算法</span>
                <select style="border: 1px solid #d9d9d9; border-radius: 4px; padding: 4px 8px; background: #fff; color: #595959; outline: none;"><option>HNSW</option></select>
            </div>
            <div style="display: flex; align-items: center; gap: 6px; font-size: 13px; color: #595959;">
                <span>实验轮次</span>
                <select style="border: 1px solid #d9d9d9; border-radius: 4px; padding: 4px 8px; background: #fff; color: #595959; outline: none;"><option>Round 1</option></select>
            </div>
            <div style="display: flex; align-items: center; gap: 6px; font-size: 13px; color: #595959;">
                <span>批次/日期</span>
                <div style="border: 1px solid #d9d9d9; border-radius: 4px; padding: 4px 8px; background: #fff; display: flex; align-items: center; gap: 4px; color: #595959;">2025-05-20 📅</div>
            </div>
            <div style="display: flex; gap: 8px; margin-left: 8px;">
                <button style="border: 1px solid #d9d9d9; background: #fff; color: #1890ff; padding: 4px 16px; border-radius: 4px; cursor: pointer; font-size: 13px;">↻ 刷新</button>
                <button style="border: none; background: #1890ff; color: #fff; padding: 4px 16px; border-radius: 4px; cursor: pointer; font-size: 13px;">⬇ 导出</button>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    kpi_html = """
    <div style="background: #fff; border: 1px solid #d9d9d9; border-radius: 8px; display: flex; align-items: center; padding: 20px 0; margin-bottom: 24px;">
        <div style="flex: 1; padding: 0 24px; border-right: 1px solid #f0f0f0;">
            <div style="font-size: 14px; color: #595959; margin-bottom: 8px; display: flex; align-items: center; gap: 4px;">查询ID</div>
            <div style="font-size: 26px; font-weight: 500; color: #262626; margin-bottom: 4px;">Q-000742 <span style="color:#1890ff; font-size:20px; float:right;">📄</span></div>
            <div style="font-size: 13px; color: #8c8c8c;">2025-05-20 14:35:22</div>
        </div>
        <div style="flex: 1; padding: 0 24px; border-right: 1px solid #f0f0f0;">
            <div style="font-size: 14px; color: #595959; margin-bottom: 8px;">Top-k</div>
            <div style="font-size: 26px; font-weight: 500; color: #262626; margin-bottom: 4px;">10 <span style="color:#bfbfbf; font-size:20px; float:right;">📊</span></div>
            <div style="font-size: 13px; color: #8c8c8c;">目标结果数</div>
        </div>
        <div style="flex: 1; padding: 0 24px; border-right: 1px solid #f0f0f0;">
            <div style="font-size: 14px; color: #595959; margin-bottom: 8px;">访问节点</div>
            <div style="font-size: 26px; font-weight: 500; color: #262626; margin-bottom: 4px;">386 <span style="color:#1890ff; font-size:20px; float:right;">📈</span></div>
            <div style="font-size: 13px; color: #8c8c8c;">单次查询访问总数</div>
        </div>
        <div style="flex: 1; padding: 0 24px; border-right: 1px solid #f0f0f0;">
            <div style="font-size: 14px; color: #595959; margin-bottom: 8px;">候选队列峰值</div>
            <div style="font-size: 26px; font-weight: 500; color: #262626; margin-bottom: 4px;">100 <span style="color:#1890ff; font-size:20px; float:right;">📊</span></div>
            <div style="font-size: 13px; color: #8c8c8c;">最大队列长度</div>
        </div>
        <div style="flex: 1; padding: 0 24px; border-right: 1px solid #f0f0f0;">
            <div style="font-size: 14px; color: #595959; margin-bottom: 8px;">过滤命中率</div>
            <div style="font-size: 26px; font-weight: 500; color: #262626; margin-bottom: 4px;">11.8% <span style="color:#1890ff; font-size:20px; float:right;">🎯</span></div>
            <div style="font-size: 13px; color: #8c8c8c;">命中 / 访问节点</div>
        </div>
        <div style="flex: 1; padding: 0 24px;">
            <div style="font-size: 14px; color: #595959; margin-bottom: 8px;">单查询延迟</div>
            <div style="font-size: 26px; font-weight: 500; color: #262626; margin-bottom: 4px;">21.6<span style="font-size: 16px;">ms</span> <span style="color:#1890ff; font-size:20px; float:right;">⏱</span></div>
            <div style="font-size: 13px; color: #8c8c8c;">P50 延迟</div>
        </div>
    </div>
    """
    kpi_html = "".join([line.strip() for line in kpi_html.split("\n")])
    st.markdown(kpi_html, unsafe_allow_html=True)

    col1, col2 = st.columns([1.3, 1])

    with col1:
        st.markdown('<div class="custom-card"><div class="card-title">HNSW 单查询路径视图 <span style="font-size:12px; color:#8c8c8c; font-weight:normal; margin-left:16px;">⚪ 入口点 ━ 搜索路径 ⚫ 普通节点 🟢 Top-k 结果 ╌ 层间连接</span></div>', unsafe_allow_html=True)
        # Placeholder for HNSW graph
        hnsw_graph_html = """
        <div style="height: 380px; position: relative; border: 1px solid #f0f0f0; border-radius: 4px; padding: 16px; background: #fff; display: flex; align-items: center; justify-content: center; margin-bottom: 8px;">
            <div style="color: #bfbfbf;">HNSW 路径图可视化展示区 (待接入真实数据渲染)</div>
        </div>
        <div style="font-size: 12px; color: #8c8c8c;">提示：拖拽可旋转视图，滚轮缩放，点击节点查看详情 <button style="float:right; border:1px solid #d9d9d9; background:#fff; border-radius:4px; padding:2px 8px; color:#1890ff; cursor:pointer;">↻ 重置视图</button></div>
        """
        st.markdown(hnsw_graph_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        
        st.markdown('<div class="custom-card"><div class="card-title">单查询 Trace 明细 <span class="info-icon">ⓘ</span></div>', unsafe_allow_html=True)
        trace_table = """
        <table style="width: 100%; border-collapse: collapse; font-size: 13px; text-align: center; margin-bottom: 16px;">
            <tr style="background: #fafafa; color: #595959;">
                <th style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0;">Step</th>
                <th style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0;">Node</th>
                <th style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0;">Layer</th>
                <th style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0;">Dist</th>
                <th style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0;">Filter</th>
                <th style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0;">Action</th>
                <th style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0;">说明</th>
            </tr>
            <tr><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0;">1</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0;">8421</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0;">3</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0;">0.812</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; color: #52c41a;">pass</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0;">entry point</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; color: #8c8c8c;">入口点</td></tr>
            <tr><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0;">2</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0;">5130</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0;">2</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0;">0.742</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; color: #ff4d4f;">fail</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0;">visited only</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; color: #8c8c8c;">未通过过滤</td></tr>
            <tr><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0;">3</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0;">2034</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0;">1</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0;">0.681</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; color: #52c41a;">pass</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0;">push candidate</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; color: #8c8c8c;">加入候选队列</td></tr>
            <tr><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0;">4</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0;">9912</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0;">0</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0;">0.634</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; color: #52c41a;">pass</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0;">update top-k</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; color: #8c8c8c;">更新 Top-k</td></tr>
            <tr><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0;">5</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0;">3107</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0;">0</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0;">0.629</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; color: #52c41a;">pass</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0;">final result</td><td style="padding: 12px 8px; border-bottom: 1px solid #f0f0f0; color: #8c8c8c;">最终结果</td></tr>
        </table>
        <div style="display: flex; justify-content: space-between; align-items: center; font-size: 13px; color: #595959;">
            <div>共 386 步</div>
            <div style="display: flex; gap: 8px; align-items: center;">
                <select style="border: 1px solid #d9d9d9; border-radius: 4px; padding: 2px 4px; background: #fff; outline: none; color: #595959;">
                    <option>10 条/页</option>
                </select>
                <div style="display: flex; gap: 4px;">
                    <div style="border: 1px solid #d9d9d9; padding: 2px 8px; border-radius: 4px; cursor: pointer; color: #bfbfbf;">&lt;</div>
                    <div style="border: 1px solid #1890ff; background: #1890ff; color: #fff; padding: 2px 8px; border-radius: 4px; cursor: pointer;">1</div>
                    <div style="border: 1px solid #d9d9d9; background: #fff; padding: 2px 8px; border-radius: 4px; cursor: pointer;">2</div>
                    <div style="border: 1px solid #d9d9d9; background: #fff; padding: 2px 8px; border-radius: 4px; cursor: pointer;">3</div>
                    <div style="border: 1px solid #d9d9d9; background: #fff; padding: 2px 8px; border-radius: 4px; cursor: pointer;">4</div>
                    <div style="border: 1px solid #d9d9d9; background: #fff; padding: 2px 8px; border-radius: 4px; cursor: pointer;">5</div>
                    <div style="color: #bfbfbf;">...</div>
                    <div style="border: 1px solid #d9d9d9; background: #fff; padding: 2px 8px; border-radius: 4px; cursor: pointer;">39</div>
                    <div style="border: 1px solid #d9d9d9; padding: 2px 8px; border-radius: 4px; cursor: pointer;">&gt;</div>
                </div>
            </div>
        </div>
        """
        trace_table = "".join([line.strip() for line in trace_table.split("\n")])
        st.markdown(trace_table, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="custom-card"><div class="card-title">候选队列与结果集变化</div>', unsafe_allow_html=True)
        fig_q = go.Figure()
        x = np.arange(0, 101, 5)
        y_queue = 120 * np.exp(-((x - 35) ** 2) / 1000)
        y_topk = 30 * (1 - np.exp(-x / 20))
        fig_q.add_trace(go.Scatter(x=x, y=y_queue, mode='lines+markers', name='Candidate Queue', line=dict(color='#1890ff', width=2), fill='tozeroy', fillcolor='rgba(24,144,255,0.1)'))
        fig_q.add_trace(go.Scatter(x=x, y=y_topk, mode='lines+markers', name='Top-k Buffer', line=dict(color='#91d5ff', width=2)))
        fig_q.update_layout(height=240, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)), xaxis_title="迭代步", yaxis_title="队列大小", plot_bgcolor="#fff")
        fig_q.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
        fig_q.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
        st.plotly_chart(fig_q, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        
        st.markdown('<div class="custom-card"><div class="card-title">过滤评估过程<div class="subtitle">基于 Bitmap 过滤，仅保留满足过滤条件的节点进入候选队列</div></div>', unsafe_allow_html=True)
        funnel_html = """
        <div style="font-size: 13px; color: #595959; margin-top: 16px;">
            <div style="display: flex; align-items: center; margin-bottom: 16px;">
                <div style="width: 60px;">访问节点</div>
                <div style="width: 40px; font-weight: 500; color: #262626;">386</div>
                <div style="flex: 1; background: #e6f7ff; height: 16px; border-radius: 8px; position: relative;">
                    <div style="position: absolute; left: 0; top: 0; height: 100%; width: 100%; background: #1890ff; border-radius: 8px;"></div>
                </div>
                <div style="width: 50px; text-align: right; font-weight: 500;">100%</div>
            </div>
            <div style="display: flex; align-items: center; margin-bottom: 16px;">
                <div style="width: 60px;">通过过滤</div>
                <div style="width: 40px; font-weight: 500; color: #262626;">142</div>
                <div style="flex: 1; background: #e6f7ff; height: 16px; border-radius: 8px; position: relative; display: flex; justify-content: center;">
                    <div style="height: 100%; width: 36.8%; background: #69b1ff; border-radius: 8px;"></div>
                </div>
                <div style="width: 50px; text-align: right; font-weight: 500;">36.8%</div>
            </div>
            <div style="display: flex; align-items: center; margin-bottom: 16px;">
                <div style="width: 60px;">候选</div>
                <div style="width: 40px; font-weight: 500; color: #262626;">78</div>
                <div style="flex: 1; background: #e6f7ff; height: 16px; border-radius: 8px; position: relative; display: flex; justify-content: center;">
                    <div style="height: 100%; width: 20.2%; background: #91d5ff; border-radius: 8px;"></div>
                </div>
                <div style="width: 50px; text-align: right; font-weight: 500;">20.2%</div>
            </div>
            <div style="display: flex; align-items: center; margin-bottom: 16px;">
                <div style="width: 60px;">Top-k</div>
                <div style="width: 40px; font-weight: 500; color: #262626;">10</div>
                <div style="flex: 1; background: #e6f7ff; height: 16px; border-radius: 8px; position: relative; display: flex; justify-content: center;">
                    <div style="height: 100%; width: 2.6%; background: #bae0ff; border-radius: 8px;"></div>
                </div>
                <div style="width: 50px; text-align: right; font-weight: 500;">2.6%</div>
            </div>
        </div>
        """
        funnel_html = "".join([line.strip() for line in funnel_html.split("\n")])
        st.markdown(funnel_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        
        st.markdown('<div class="custom-card"><div class="card-title">查询策略对比</div>', unsafe_allow_html=True)
        fig_c = make_subplots(specs=[[{"secondary_y": True}]])
        strategies = ['Post-Filter', 'WST', 'SIEVE', 'Optimized HNSW']
        latency = [96.8, 58.3, 29.7, 21.6]
        recall = [0.691, 0.824, 0.932, 0.967]
        
        fig_c.add_trace(go.Bar(x=strategies, y=latency, name="延迟 (ms)", marker_color="#1890ff", width=0.3, text=latency, textposition='outside', textfont=dict(size=11)), secondary_y=False)
        fig_c.add_trace(go.Scatter(x=strategies, y=recall, name="召回率 (Recall@10)", mode='lines+markers+text', line=dict(color="#69b1ff", width=2), marker=dict(symbol="circle", size=8), text=[f"{r*100:.1f}%" for r in recall], textposition='top center', textfont=dict(size=11)), secondary_y=True)
        
        fig_c.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor="#fff", legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5, font=dict(size=10)))
        fig_c.update_yaxes(title_text="延迟 (ms)", range=[0, 130], showgrid=True, gridcolor="#f0f0f0", secondary_y=False)
        fig_c.update_yaxes(title_text="召回率", range=[0, 1.1], showgrid=False, secondary_y=True, tickformat=".0%")
        st.plotly_chart(fig_c, use_container_width=True, config={'displayModeBar': False})
        st.markdown('<div style="font-size: 12px; color: #8c8c8c; margin-top: 8px;">说明：在相同查询 Q-000742 与 Top-k=10 条件下的对比结果。</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# 页面 4: 实验报告导出 (Experiment Report Export)
# =============================================================================
def render_export_report():
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: flex-end; padding-bottom: 16px; margin-bottom: 16px;">
        <div>
            <div style="font-size: 13px; color: #8c8c8c; margin-bottom: 6px;">首页 / 实验分析 / <span style="color: #262626;">实验报告导出</span></div>
            <div style="font-size: 22px; font-weight: 600; color: #262626; margin-bottom: 4px;">实验报告导出</div>
            <div style="font-size: 13px; color: #8c8c8c;">汇总实验配置、算法输出、图表、日志和论文附录所需表格，生成可复现实验报告</div>
        </div>
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="display: flex; align-items: center; gap: 6px; font-size: 13px; color: #595959;">
                <span>数据集</span>
                <select style="border: 1px solid #d9d9d9; border-radius: 4px; padding: 4px 8px; background: #fff; color: #595959; outline: none;"><option>Deep1B</option></select>
            </div>
            <div style="display: flex; align-items: center; gap: 6px; font-size: 13px; color: #595959;">
                <span>算法</span>
                <select style="border: 1px solid #d9d9d9; border-radius: 4px; padding: 4px 8px; background: #fff; color: #595959; outline: none;"><option>HNSW</option></select>
            </div>
            <div style="display: flex; align-items: center; gap: 6px; font-size: 13px; color: #595959;">
                <span>实验轮次</span>
                <select style="border: 1px solid #d9d9d9; border-radius: 4px; padding: 4px 8px; background: #fff; color: #595959; outline: none;"><option>Round 1</option></select>
            </div>
            <div style="display: flex; align-items: center; gap: 6px; font-size: 13px; color: #595959;">
                <span>批次/日期</span>
                <div style="border: 1px solid #d9d9d9; border-radius: 4px; padding: 4px 8px; background: #fff; display: flex; align-items: center; gap: 4px; color: #595959;">2025-05-20 📅</div>
            </div>
            <div style="display: flex; gap: 8px; margin-left: 8px;">
                <button style="border: 1px solid #d9d9d9; background: #fff; color: #1890ff; padding: 4px 16px; border-radius: 4px; cursor: pointer; font-size: 13px;">↻ 刷新</button>
                <button style="border: none; background: #1890ff; color: #fff; padding: 4px 16px; border-radius: 4px; cursor: pointer; font-size: 13px;">⬇ 导出</button>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    kpi_html = """
    <div style="background: #fff; border: 1px solid #d9d9d9; border-radius: 8px; display: flex; align-items: center; padding: 20px 0; margin-bottom: 24px;">
        <div style="flex: 1; padding: 0 24px; border-right: 1px solid #f0f0f0;">
            <div style="font-size: 14px; color: #595959; margin-bottom: 8px;">报告模板</div>
            <div style="font-size: 26px; font-weight: 500; color: #262626; margin-bottom: 4px;">4<span style="font-size:16px;">个</span> <span style="color:#1890ff; font-size:20px; float:right;">📄</span></div>
            <div style="font-size: 13px; color: #8c8c8c;">模板可用</div>
        </div>
        <div style="flex: 1; padding: 0 24px; border-right: 1px solid #f0f0f0;">
            <div style="font-size: 14px; color: #595959; margin-bottom: 8px;">已选图表</div>
            <div style="font-size: 26px; font-weight: 500; color: #262626; margin-bottom: 4px;">18<span style="font-size:16px;">个</span> <span style="color:#1890ff; font-size:20px; float:right;">📊</span></div>
            <div style="font-size: 13px; color: #8c8c8c;"><span style="color: #1890ff;">↑ 2</span> 较上轮</div>
        </div>
        <div style="flex: 1; padding: 0 24px; border-right: 1px solid #f0f0f0;">
            <div style="font-size: 14px; color: #595959; margin-bottom: 8px;">数据表</div>
            <div style="font-size: 26px; font-weight: 500; color: #262626; margin-bottom: 4px;">7<span style="font-size:16px;">张</span> <span style="color:#1890ff; font-size:20px; float:right;">📋</span></div>
            <div style="font-size: 13px; color: #8c8c8c;"><span style="color: #1890ff;">↑ 1</span> 较上轮</div>
        </div>
        <div style="flex: 1; padding: 0 24px; border-right: 1px solid #f0f0f0;">
            <div style="font-size: 14px; color: #595959; margin-bottom: 8px;">日志文件</div>
            <div style="font-size: 26px; font-weight: 500; color: #262626; margin-bottom: 4px;">12<span style="font-size:16px;">个</span> <span style="color:#1890ff; font-size:20px; float:right;">📝</span></div>
            <div style="font-size: 13px; color: #8c8c8c;"><span style="color: #1890ff;">↑ 3</span> 较上轮</div>
        </div>
        <div style="flex: 1; padding: 0 24px; border-right: 1px solid #f0f0f0;">
            <div style="font-size: 14px; color: #595959; margin-bottom: 8px;">可复现资产</div>
            <div style="font-size: 26px; font-weight: 500; color: #262626; margin-bottom: 4px;">92.6% <span style="color:#1890ff; font-size:20px; float:right;">📦</span></div>
            <div style="font-size: 13px; color: #8c8c8c;"><span style="color: #1890ff;">↑ 2.1%</span> 较上轮</div>
        </div>
        <div style="flex: 1; padding: 0 24px;">
            <div style="font-size: 14px; color: #595959; margin-bottom: 8px;">导出状态</div>
            <div style="font-size: 26px; font-weight: 500; color: #262626; margin-bottom: 4px;">就绪 <span style="color:#52c41a; font-size:20px; float:right;">✓</span></div>
            <div style="font-size: 13px; color: #8c8c8c;">可立即导出</div>
        </div>
    </div>
    """
    kpi_html = "".join([line.strip() for line in kpi_html.split("\n")])
    st.markdown(kpi_html, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.5, 1])

    with col1:
        st.markdown('<div class="custom-card"><div class="card-title">报告内容编排</div>', unsafe_allow_html=True)
        content_html = """
        <div style="display: flex; flex-direction: column; gap: 12px; margin-top: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px; border: 1px solid #e8e8e8; border-radius: 4px; background: #fafafa;">
                <div><input type="checkbox" checked style="margin-right: 8px;"> 实验总览</div>
                <div style="font-size: 12px; color: #52c41a; background: #f6ffed; border: 1px solid #b7eb8f; padding: 2px 6px; border-radius: 4px;">已选</div>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px; border: 1px solid #e8e8e8; border-radius: 4px; background: #fafafa;">
                <div><input type="checkbox" checked style="margin-right: 8px;"> 数据集管理</div>
                <div style="font-size: 12px; color: #52c41a; background: #f6ffed; border: 1px solid #b7eb8f; padding: 2px 6px; border-radius: 4px;">已选</div>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px; border: 1px solid #e8e8e8; border-radius: 4px; background: #fafafa;">
                <div><input type="checkbox" checked style="margin-right: 8px;"> 过滤条件</div>
                <div style="font-size: 12px; color: #52c41a; background: #f6ffed; border: 1px solid #b7eb8f; padding: 2px 6px; border-radius: 4px;">已选</div>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px; border: 1px solid #e8e8e8; border-radius: 4px; background: #fafafa;">
                <div><input type="checkbox" checked style="margin-right: 8px;"> HNSW构建</div>
                <div style="font-size: 12px; color: #52c41a; background: #f6ffed; border: 1px solid #b7eb8f; padding: 2px 6px; border-radius: 4px;">已选</div>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px; border: 1px solid #e8e8e8; border-radius: 4px; background: #fafafa;">
                <div><input type="checkbox" checked style="margin-right: 8px;"> 性能对比</div>
                <div style="font-size: 12px; color: #52c41a; background: #f6ffed; border: 1px solid #b7eb8f; padding: 2px 6px; border-radius: 4px;">已选</div>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px; border: 1px solid #e8e8e8; border-radius: 4px; background: #fafafa;">
                <div><input type="checkbox" checked style="margin-right: 8px;"> 参数敏感性</div>
                <div style="font-size: 12px; color: #52c41a; background: #f6ffed; border: 1px solid #b7eb8f; padding: 2px 6px; border-radius: 4px;">已选</div>
            </div>
            <button style="width: 100%; background: #1890ff; color: white; border: none; border-radius: 4px; padding: 8px 0; margin-top: 8px; cursor: pointer; font-size: 14px; display: flex; align-items: center; justify-content: center; gap: 8px;">
                📄 生成报告
            </button>
        </div>
        """
        content_html = "".join([line.strip() for line in content_html.split("\n")])
        st.markdown(content_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="custom-card"><div class="card-title">报告预览 <span style="float:right; color:#1890ff; cursor:pointer;">⤢</span></div>', unsafe_allow_html=True)
        preview_html = """
        <div style="border: 1px solid #f0f0f0; border-radius: 4px; padding: 20px; background: #fafafa; height: 320px; display: flex; flex-direction: column; align-items: center;">
            <div style="font-size: 18px; font-weight: 600; margin-bottom: 16px; color: #262626;">过滤向量检索优化实验报告</div>
            <div style="width: 100%; display: flex; gap: 16px; margin-bottom: 16px;">
                <div style="flex: 1; border: 1px solid #e8e8e8; background: #fff; padding: 12px; border-radius: 4px;">
                    <div style="font-size: 12px; color: #8c8c8c;">实验概览</div>
                    <div style="display: flex; justify-content: space-between; margin-top: 8px;">
                        <div><div style="font-size: 11px; color: #bfbfbf;">数据集</div><div style="font-size: 12px;">Deep1B</div></div>
                        <div><div style="font-size: 11px; color: #bfbfbf;">算法</div><div style="font-size: 12px;">HNSW</div></div>
                        <div><div style="font-size: 11px; color: #bfbfbf;">实验轮次</div><div style="font-size: 12px;">Round 1</div></div>
                        <div><div style="font-size: 11px; color: #bfbfbf;">生成时间</div><div style="font-size: 12px;">2025-05-20</div></div>
                    </div>
                </div>
            </div>
            <div style="width: 100%; display: flex; justify-content: space-between; margin-bottom: 16px;">
                <div style="text-align: center;"><div style="font-size: 12px; color: #8c8c8c;">Recall@10</div><div style="font-size: 16px; font-weight: 500;">0.926</div></div>
                <div style="text-align: center;"><div style="font-size: 12px; color: #8c8c8c;">查询延迟</div><div style="font-size: 16px; font-weight: 500;">24.532 <span style="font-size:10px;">ms</span></div></div>
                <div style="text-align: center;"><div style="font-size: 12px; color: #8c8c8c;">内存占用</div><div style="font-size: 16px; font-weight: 500;">3.45 <span style="font-size:10px;">GB</span></div></div>
                <div style="text-align: center;"><div style="font-size: 12px; color: #8c8c8c;">索引规模</div><div style="font-size: 16px; font-weight: 500;">1.02 <span style="font-size:10px;">TB</span></div></div>
            </div>
            <div style="width: 100%; flex: 1; border: 1px solid #e8e8e8; background: #fff; padding: 8px; border-radius: 4px; display: flex; flex-direction: column;">
                <div style="font-size: 12px; color: #595959; margin-bottom: 8px;">性能趋势 (Recall@10)</div>
                <div style="flex: 1; position: relative;">
                    <svg width="100%" height="100%" preserveAspectRatio="none">
                        <polyline points="0,60 30,50 60,30 90,40 120,20 150,25 180,10 210,15 240,5 270,10 300,5" fill="none" stroke="#1890ff" stroke-width="2" />
                        <circle cx="0" cy="60" r="3" fill="#1890ff"/> <circle cx="30" cy="50" r="3" fill="#1890ff"/> <circle cx="60" cy="30" r="3" fill="#1890ff"/>
                        <circle cx="90" cy="40" r="3" fill="#1890ff"/> <circle cx="120" cy="20" r="3" fill="#1890ff"/> <circle cx="150" cy="25" r="3" fill="#1890ff"/>
                        <circle cx="180" cy="10" r="3" fill="#1890ff"/> <circle cx="210" cy="15" r="3" fill="#1890ff"/> <circle cx="240" cy="5" r="3" fill="#1890ff"/>
                        <circle cx="270" cy="10" r="3" fill="#1890ff"/> <circle cx="300" cy="5" r="3" fill="#1890ff"/>
                    </svg>
                </div>
            </div>
            <div style="margin-top: 12px; font-size: 12px; color: #8c8c8c;">&lt; &lt; 1 / 12 &gt; &gt;</div>
        </div>
        """
        preview_html = "".join([line.strip() for line in preview_html.split("\n")])
        st.markdown(preview_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="custom-card"><div class="card-title">导出格式与资产包</div>', unsafe_allow_html=True)
        formats_html = """
        <div style="display: flex; flex-direction: column; gap: 12px; margin-top: 16px;">
            <div style="border: 1px solid #1890ff; border-radius: 4px; padding: 12px; background: #e6f7ff; display: flex; align-items: flex-start; gap: 12px; cursor: pointer;">
                <div style="background: #ff4d4f; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 12px;">PDF</div>
                <div>
                    <div style="font-size: 14px; font-weight: 500; color: #262626; display: flex; justify-content: space-between;">PDF <span style="font-size: 11px; background: #1890ff; color: white; padding: 2px 6px; border-radius: 10px; font-weight: normal;">推荐</span></div>
                    <div style="font-size: 12px; color: #595959; margin-top: 4px;">生成排版精美的实验报告（推荐用于论文与归档）</div>
                </div>
            </div>
            <div style="border: 1px solid #e8e8e8; border-radius: 4px; padding: 12px; background: #fff; display: flex; align-items: flex-start; gap: 12px; cursor: pointer;">
                <div style="background: #ff7a45; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 12px;">HTML</div>
                <div>
                    <div style="font-size: 14px; font-weight: 500; color: #262626;">HTML</div>
                    <div style="font-size: 12px; color: #595959; margin-top: 4px;">交互式网页报告，便于在线浏览与分享</div>
                </div>
            </div>
            <div style="border: 1px solid #e8e8e8; border-radius: 4px; padding: 12px; background: #fff; display: flex; align-items: flex-start; gap: 12px; cursor: pointer;">
                <div style="background: #52c41a; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 12px;">XLSX</div>
                <div>
                    <div style="font-size: 14px; font-weight: 500; color: #262626;">XLSX</div>
                    <div style="font-size: 12px; color: #595959; margin-top: 4px;">数据表与汇总表导出，便于二次分析</div>
                </div>
            </div>
            <div style="border: 1px solid #e8e8e8; border-radius: 4px; padding: 12px; background: #fff; display: flex; align-items: flex-start; gap: 12px; cursor: pointer;">
                <div style="background: #722ed1; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 12px;">ZIP</div>
                <div>
                    <div style="font-size: 14px; font-weight: 500; color: #262626;">ZIP</div>
                    <div style="font-size: 12px; color: #595959; margin-top: 4px;">完整资产包：配置、数据、图表、日志与代码快照</div>
                </div>
            </div>
        </div>
        """
        formats_html = "".join([line.strip() for line in formats_html.split("\n")])
        st.markdown(formats_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    
    col4, col5 = st.columns([1, 1])
    
    with col4:
        st.markdown('<div class="custom-card"><div class="card-title">可复现实验清单 <span class="info-icon">ⓘ</span></div>', unsafe_allow_html=True)
        checklist_html = """
        <table style="width: 100%; border-collapse: collapse; font-size: 13px; text-align: left; margin-top: 8px;">
            <tr style="background: #fafafa; color: #595959;">
                <th style="padding: 10px 12px; border-bottom: 1px solid #f0f0f0;">检查项</th>
                <th style="padding: 10px 12px; border-bottom: 1px solid #f0f0f0;">说明</th>
                <th style="padding: 10px 12px; border-bottom: 1px solid #f0f0f0;">状态</th>
                <th style="padding: 10px 12px; border-bottom: 1px solid #f0f0f0;">完整度</th>
            </tr>
            <tr>
                <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;"><span style="color:#1890ff;">✓</span> 代码版本</td>
                <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #595959;">代码仓库与提交哈希一致</td>
                <td style="padding: 12px; border-bottom: 1px solid #f0f0f0;"><span style="color: #52c41a;">通过</span></td>
                <td style="padding: 12px; border-bottom: 1px solid #f0f0f0;"><div style="display:flex; align-items:center; gap:8px;"><div style="flex:1; height:6px; background:#e8e8e8; border-radius:3px;"><div style="width:100%; height:100%; background:#52c41a; border-radius:3px;"></div></div>100%</div></td>
            </tr>
            <tr>
                <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;"><span style="color:#1890ff;">✓</span> 数据版本</td>
                <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #595959;">数据集版本号校验一致</td>
                <td style="padding: 12px; border-bottom: 1px solid #f0f0f0;"><span style="color: #52c41a;">通过</span></td>
                <td style="padding: 12px; border-bottom: 1px solid #f0f0f0;"><div style="display:flex; align-items:center; gap:8px;"><div style="flex:1; height:6px; background:#e8e8e8; border-radius:3px;"><div style="width:100%; height:100%; background:#52c41a; border-radius:3px;"></div></div>100%</div></td>
            </tr>
            <tr>
                <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;"><span style="color:#1890ff;">✓</span> 参数配置</td>
                <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #595959;">实验参数文件完整且可加载</td>
                <td style="padding: 12px; border-bottom: 1px solid #f0f0f0;"><span style="color: #52c41a;">通过</span></td>
                <td style="padding: 12px; border-bottom: 1px solid #f0f0f0;"><div style="display:flex; align-items:center; gap:8px;"><div style="flex:1; height:6px; background:#e8e8e8; border-radius:3px;"><div style="width:100%; height:100%; background:#52c41a; border-radius:3px;"></div></div>100%</div></td>
            </tr>
            <tr>
                <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;"><span style="color:#1890ff;">✓</span> 原始输出</td>
                <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #595959;">算法输出文件完整性校验</td>
                <td style="padding: 12px; border-bottom: 1px solid #f0f0f0;"><span style="color: #faad14;">警告</span></td>
                <td style="padding: 12px; border-bottom: 1px solid #f0f0f0;"><div style="display:flex; align-items:center; gap:8px;"><div style="flex:1; height:6px; background:#e8e8e8; border-radius:3px;"><div style="width:85%; height:100%; background:#faad14; border-radius:3px;"></div></div>85%</div></td>
            </tr>
            <tr>
                <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;"><span style="color:#1890ff;">✓</span> 环境信息</td>
                <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #595959;">运行环境与依赖可复现</td>
                <td style="padding: 12px; border-bottom: 1px solid #f0f0f0;"><span style="color: #52c41a;">通过</span></td>
                <td style="padding: 12px; border-bottom: 1px solid #f0f0f0;"><div style="display:flex; align-items:center; gap:8px;"><div style="flex:1; height:6px; background:#e8e8e8; border-radius:3px;"><div style="width:100%; height:100%; background:#52c41a; border-radius:3px;"></div></div>100%</div></td>
            </tr>
        </table>
        """
        checklist_html = "".join([line.strip() for line in checklist_html.split("\n")])
        st.markdown(checklist_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col5:
        st.markdown('<div class="custom-card"><div class="card-title">最近导出记录</div>', unsafe_allow_html=True)
        history_html = """
        <table style="width: 100%; border-collapse: collapse; font-size: 13px; text-align: left; margin-top: 8px;">
            <tr style="background: #fafafa; color: #595959;">
                <th style="padding: 10px 12px; border-bottom: 1px solid #f0f0f0;">时间</th>
                <th style="padding: 10px 12px; border-bottom: 1px solid #f0f0f0;">类型</th>
                <th style="padding: 10px 12px; border-bottom: 1px solid #f0f0f0;">文件名</th>
                <th style="padding: 10px 12px; border-bottom: 1px solid #f0f0f0;">大小</th>
                <th style="padding: 10px 12px; border-bottom: 1px solid #f0f0f0;">状态</th>
                <th style="padding: 10px 12px; border-bottom: 1px solid #f0f0f0;">操作</th>
            </tr>
            <tr>
                <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #595959;">2024-05-20 23:10</td>
                <td style="padding: 12px; border-bottom: 1px solid #f0f0f0;"><span style="color: #ff4d4f; border: 1px solid #ff4d4f; padding: 2px 6px; border-radius: 4px; font-size: 10px;">PDF</span></td>
                <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">filtered_ann_report_v3.pdf</td>
                <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #595959;">4.8MB</td>
                <td style="padding: 12px; border-bottom: 1px solid #f0f0f0;"><span class="badge-success">完成</span></td>
                <td style="padding: 12px; border-bottom: 1px solid #f0f0f0;"><span style="color: #1890ff; cursor: pointer; margin-right: 8px;">⬇</span><span style="color: #8c8c8c; cursor: pointer;">...</span></td>
            </tr>
            <tr>
                <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #595959;">2024-05-20 22:47</td>
                <td style="padding: 12px; border-bottom: 1px solid #f0f0f0;"><span style="color: #ff7a45; border: 1px solid #ff7a45; padding: 2px 6px; border-radius: 4px; font-size: 10px;">HTML</span></td>
                <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">filtered_ann_report_v3.html</td>
                <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #595959;">6.2MB</td>
                <td style="padding: 12px; border-bottom: 1px solid #f0f0f0;"><span class="badge-success">完成</span></td>
                <td style="padding: 12px; border-bottom: 1px solid #f0f0f0;"><span style="color: #1890ff; cursor: pointer; margin-right: 8px;">⬇</span><span style="color: #8c8c8c; cursor: pointer;">...</span></td>
            </tr>
            <tr>
                <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #595959;">2024-05-20 22:15</td>
                <td style="padding: 12px; border-bottom: 1px solid #f0f0f0;"><span style="color: #722ed1; border: 1px solid #722ed1; padding: 2px 6px; border-radius: 4px; font-size: 10px;">ZIP</span></td>
                <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">filtered_ann_report_v3_assets.zip</td>
                <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #595959;">128.7MB</td>
                <td style="padding: 12px; border-bottom: 1px solid #f0f0f0;"><span class="badge-success">完成</span></td>
                <td style="padding: 12px; border-bottom: 1px solid #f0f0f0;"><span style="color: #1890ff; cursor: pointer; margin-right: 8px;">⬇</span><span style="color: #8c8c8c; cursor: pointer;">...</span></td>
            </tr>
        </table>
        """
        history_html = "".join([line.strip() for line in history_html.split("\n")])
        st.markdown(history_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    
    st.markdown('<div class="custom-card"><div class="card-title">报告图表与数据源明细 <span class="info-icon">ⓘ</span></div>', unsafe_allow_html=True)
    details_html = """
    <table style="width: 100%; border-collapse: collapse; font-size: 13px; text-align: center; margin-top: 8px;">
        <tr style="background: #fafafa; color: #595959;">
            <th style="padding: 12px; border-bottom: 1px solid #f0f0f0;">模块</th>
            <th style="padding: 12px; border-bottom: 1px solid #f0f0f0;">图表/表格</th>
            <th style="padding: 12px; border-bottom: 1px solid #f0f0f0;">数据源</th>
            <th style="padding: 12px; border-bottom: 1px solid #f0f0f0;">输出文件</th>
            <th style="padding: 12px; border-bottom: 1px solid #f0f0f0;">状态</th>
            <th style="padding: 12px; border-bottom: 1px solid #f0f0f0;">备注</th>
            <th style="padding: 12px; border-bottom: 1px solid #f0f0f0;">操作</th>
        </tr>
        <tr>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #595959;">实验总览</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">Recall-QPS 曲线</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #595959;">results/normalized/all_results.csv</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #595959;">figures/recall_qps.png</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0;"><span style="color: #52c41a;">● 完成</span></td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #595959;">主图</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0;"><span style="color: #1890ff; cursor: pointer;">👁</span></td>
        </tr>
        <tr>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #595959;">数据集管理</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">文件版本表</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #595959;">data/processed+results/raw</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #595959;">tables/dataset_files.xlsx</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0;"><span style="color: #52c41a;">● 完成</span></td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #595959;">附录</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0;"><span style="color: #1890ff; cursor: pointer;">👁</span></td>
        </tr>
        <tr>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #595959;">查询可视化</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">单查询 Trace</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #595959;">results/raw/sieve_trace.csv</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #595959;">figures/query_trace.png</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0;"><span style="color: #52c41a;">● 完成</span></td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #595959;">案例</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0;"><span style="color: #1890ff; cursor: pointer;">👁</span></td>
        </tr>
        <tr>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #595959;">性能对比</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">算法对比表</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #595959;">all_results.csv</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #595959;">tables/algorithm_compare.xlsx</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0;"><span style="color: #52c41a;">● 完成</span></td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #595959;">论文表格</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0;"><span style="color: #1890ff; cursor: pointer;">👁</span></td>
        </tr>
        <tr>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #595959;">参数敏感性</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">M x efSearch 热力图</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #595959;">experiments/param_grid.csv</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #595959;">figures/param_heatmap.png</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0;"><span style="color: #52c41a;">● 完成</span></td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #595959;">附录</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0;"><span style="color: #1890ff; cursor: pointer;">👁</span></td>
        </tr>
    </table>
    """
    details_html = "".join([line.strip() for line in details_html.split("\n")])
    st.markdown(details_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# 页面 5: 性能对比分析 (Performance Comparison)
# =============================================================================
def render_perf_compare():
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: flex-end; padding-bottom: 16px; margin-bottom: 16px;">
        <div>
            <div style="font-size: 13px; color: #8c8c8c; margin-bottom: 6px;">首页 / 实验分析 / <span style="color: #262626;">性能对比分析</span></div>
            <div style="font-size: 22px; font-weight: 600; color: #262626; margin-bottom: 4px;">性能对比分析</div>
            <div style="font-size: 13px; color: #8c8c8c;">对比 Pre-filter、Post-filter、HNSW Filter、WST、SIEVE 与优化 HNSW 的 Recall、QPS、延迟和资源</div>
        </div>
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="display: flex; align-items: center; gap: 6px; font-size: 13px; color: #595959;">
                <span>数据集</span>
                <select style="border: 1px solid #d9d9d9; border-radius: 4px; padding: 4px 8px; background: #fff; color: #595959; outline: none;"><option>Deep1B</option></select>
            </div>
            <div style="display: flex; align-items: center; gap: 6px; font-size: 13px; color: #595959;">
                <span>查询模板</span>
                <select style="border: 1px solid #d9d9d9; border-radius: 4px; padding: 4px 8px; background: #fff; color: #595959; outline: none;"><option>全部模板</option></select>
            </div>
            <div style="display: flex; align-items: center; gap: 6px; font-size: 13px; color: #595959;">
                <span>选择率范围</span>
                <select style="border: 1px solid #d9d9d9; border-radius: 4px; padding: 4px 8px; background: #fff; color: #595959; outline: none;"><option>1% ~ 100%</option></select>
            </div>
            <div style="display: flex; align-items: center; gap: 6px; font-size: 13px; color: #595959;">
                <span>批次/日期</span>
                <div style="border: 1px solid #d9d9d9; border-radius: 4px; padding: 4px 8px; background: #fff; display: flex; align-items: center; gap: 4px; color: #595959;">2025-05-20 📅</div>
            </div>
            <div style="display: flex; gap: 8px; margin-left: 8px;">
                <button style="border: 1px solid #d9d9d9; background: #fff; color: #1890ff; padding: 4px 16px; border-radius: 4px; cursor: pointer; font-size: 13px;">↻ 刷新</button>
                <button style="border: none; background: #1890ff; color: #fff; padding: 4px 16px; border-radius: 4px; cursor: pointer; font-size: 13px;">⬇ 导出</button>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    kpi_html = """
    <div style="background: #fff; border: 1px solid #d9d9d9; border-radius: 8px; display: flex; align-items: center; padding: 20px 0; margin-bottom: 24px;">
        <div style="flex: 1; padding: 0 24px; border-right: 1px solid #f0f0f0;">
            <div style="font-size: 14px; color: #595959; margin-bottom: 8px;">最优方法</div>
            <div style="font-size: 20px; font-weight: 600; color: #262626; margin-bottom: 4px;">Optimized HNSW <span style="color:#1890ff; font-size:20px; float:right;">🏆</span></div>
            <div style="font-size: 13px; color: #8c8c8c;">综合评分第一</div>
        </div>
        <div style="flex: 1; padding: 0 24px; border-right: 1px solid #f0f0f0;">
            <div style="font-size: 14px; color: #595959; margin-bottom: 8px;">最高 QPS</div>
            <div style="font-size: 26px; font-weight: 500; color: #262626; margin-bottom: 4px;">1284 <span style="color:#1890ff; font-size:20px; float:right;">⏱</span></div>
            <div style="font-size: 13px; color: #8c8c8c;"><span style="color: #1890ff;">↑ 36.7%</span> 相比次优</div>
        </div>
        <div style="flex: 1; padding: 0 24px; border-right: 1px solid #f0f0f0;">
            <div style="font-size: 14px; color: #595959; margin-bottom: 8px;">最低 P95 延迟</div>
            <div style="font-size: 26px; font-weight: 500; color: #262626; margin-bottom: 4px;">23.8<span style="font-size: 16px;">ms</span> <span style="color:#1890ff; font-size:20px; float:right;">📉</span></div>
            <div style="font-size: 13px; color: #8c8c8c;"><span style="color: #1890ff;">↓ 31.2%</span> 相比次优</div>
        </div>
        <div style="flex: 1; padding: 0 24px; border-right: 1px solid #f0f0f0;">
            <div style="font-size: 14px; color: #595959; margin-bottom: 8px;">最佳低选择率</div>
            <div style="font-size: 22px; font-weight: 500; color: #262626; margin-bottom: 4px;">SIEVE <span style="color:#1890ff; font-size:20px; float:right;">🎯</span></div>
            <div style="font-size: 13px; color: #8c8c8c;">在 ≤10% 选择率下</div>
        </div>
        <div style="flex: 1; padding: 0 24px; border-right: 1px solid #f0f0f0;">
            <div style="font-size: 14px; color: #595959; margin-bottom: 8px;">平均内存</div>
            <div style="font-size: 26px; font-weight: 500; color: #262626; margin-bottom: 4px;">6.54<span style="font-size: 16px;">GB</span> <span style="color:#1890ff; font-size:20px; float:right;">💾</span></div>
            <div style="font-size: 13px; color: #8c8c8c;">内存占用均值</div>
        </div>
        <div style="flex: 1; padding: 0 24px;">
            <div style="font-size: 14px; color: #595959; margin-bottom: 8px;">有效实验</div>
            <div style="font-size: 26px; font-weight: 500; color: #262626; margin-bottom: 4px;">143 <span style="color:#1890ff; font-size:20px; float:right;">🔬</span></div>
            <div style="font-size: 13px; color: #8c8c8c;">组对比实验</div>
        </div>
    </div>
    """
    kpi_html = "".join([line.strip() for line in kpi_html.split("\n")])
    st.markdown(kpi_html, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="custom-card"><div class="card-title">Recall-QPS 前沿曲线 <span class="info-icon">ⓘ</span><span style="float:right; font-size:12px; color:#1890ff; font-weight:normal; border:1px solid #1890ff; padding:2px 8px; border-radius:4px;">AUC 指标对比</span></div>', unsafe_allow_html=True)
        fig1 = go.Figure()
        recalls = np.linspace(0.8, 1.0, 15)
        fig1.add_trace(go.Scatter(x=recalls, y=1000 * np.exp(-15 * (recalls - 0.8)), name="HNSW-Base", mode="lines+markers", line=dict(color="#bfbfbf", width=2), marker=dict(symbol="circle", size=5)))
        fig1.add_trace(go.Scatter(x=recalls, y=500 * np.exp(-12 * (recalls - 0.8)), name="Post-Filter", mode="lines+markers", line=dict(color="#91d5ff", width=2), marker=dict(symbol="circle", size=5)))
        fig1.add_trace(go.Scatter(x=recalls, y=800 * np.exp(-10 * (recalls - 0.8)), name="Window Search Tree", mode="lines+markers", line=dict(color="#69b1ff", width=2), marker=dict(symbol="circle", size=5)))
        fig1.add_trace(go.Scatter(x=recalls, y=1100 * np.exp(-8 * (recalls - 0.8)), name="SIEVE", mode="lines+markers", line=dict(color="#40a9ff", width=2), marker=dict(symbol="circle", size=5)))
        fig1.add_trace(go.Scatter(x=recalls, y=1500 * np.exp(-6 * (recalls - 0.8)), name="Optimized HNSW", mode="lines+markers", line=dict(color="#1890ff", width=3), marker=dict(symbol="circle", size=6)))
        
        fig1.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor="#fff", legend=dict(orientation="v", yanchor="top", y=0.9, xanchor="right", x=0.95, font=dict(size=10)))
        fig1.update_xaxes(title="Recall@10 (→)", showgrid=True, gridcolor="#f0f0f0")
        fig1.update_yaxes(title="QPS (↑)", showgrid=True, gridcolor="#f0f0f0")
        st.plotly_chart(fig1, use_container_width=True, config={'displayModeBar': False})
        st.markdown('<div style="font-size: 12px; color: #8c8c8c; margin-top: 8px;">📈 右上角为更优区域：更高 QPS 与更高 Recall</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="custom-card"><div class="card-title">选择率分段性能 (P95 延迟) <span class="info-icon">ⓘ</span></div>', unsafe_allow_html=True)
        fig2 = go.Figure()
        sel = ["1%", "5%", "10%", "25%", "50%", "100%"]
        fig2.add_trace(go.Scatter(x=sel, y=[140, 160, 180, 220, 280, 350], name="Pre-filter", mode="lines+markers", line=dict(color="#bfbfbf", width=2)))
        fig2.add_trace(go.Scatter(x=sel, y=[120, 130, 150, 190, 240, 300], name="Post-filter", mode="lines+markers", line=dict(color="#91d5ff", width=2)))
        fig2.add_trace(go.Scatter(x=sel, y=[80, 90, 110, 140, 180, 220], name="Window Search Tree", mode="lines+markers", line=dict(color="#69b1ff", width=2)))
        fig2.add_trace(go.Scatter(x=sel, y=[40, 50, 70, 100, 140, 180], name="SIEVE", mode="lines+markers", line=dict(color="#52c41a", width=2)))
        fig2.add_trace(go.Scatter(x=sel, y=[20, 25, 41.8, 70, 128.7, 248.6], name="Optimized HNSW", mode="lines+markers", line=dict(color="#1890ff", width=3)))
        
        fig2.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor="#fff", legend=dict(orientation="v", yanchor="top", y=0.9, xanchor="right", x=0.95, font=dict(size=10)))
        fig2.update_xaxes(title="选择率", showgrid=True, gridcolor="#f0f0f0")
        fig2.update_yaxes(title="P95 延迟 (ms)", showgrid=True, gridcolor="#f0f0f0", type="log")
        st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})
        st.markdown('<div style="font-size: 12px; color: #8c8c8c; margin-top: 8px; display: flex; justify-content: space-between;"><span>📉 越低越好 (对数坐标)</span><span style="color:#1890ff;">✦ 关键点已高亮标注</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    
    col3, col4, col5 = st.columns([1, 1, 1])

    with col3:
        st.markdown('<div class="custom-card"><div class="card-title">综合指标评分 <span class="info-icon">ⓘ</span></div>', unsafe_allow_html=True)
        fig3 = go.Figure()
        categories = ['Recall@10 (25%)', 'P95 延迟 (20%)', '内存占用 (15%)', '构建时间 (10%)', 'QPS (30%)']
        fig3.add_trace(go.Scatterpolar(r=[95, 90, 85, 80, 98], theta=categories, fill='toself', name='Optimized HNSW', line=dict(color="#1890ff"), fillcolor="rgba(24,144,255,0.2)"))
        fig3.add_trace(go.Scatterpolar(r=[90, 85, 80, 85, 85], theta=categories, fill='toself', name='SIEVE', line=dict(color="#52c41a"), fillcolor="rgba(82,196,26,0.1)"))
        fig3.add_trace(go.Scatterpolar(r=[85, 75, 90, 90, 70], theta=categories, fill='toself', name='WST', line=dict(color="#722ed1"), fillcolor="rgba(114,46,209,0.1)"))
        
        fig3.update_layout(
            height=220, margin=dict(l=30, r=30, t=20, b=20),
            polar=dict(radialaxis=dict(visible=True, range=[0, 100], showticklabels=False)),
            showlegend=False
        )
        st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar': False})
        
        score_html = """
        <div style="font-size: 12px; margin-top: 8px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px; color: #1890ff; font-weight: 600;"><span>1. Optimized HNSW</span><span>92.4</span></div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px; color: #52c41a;"><span>2. SIEVE</span><span>76.8</span></div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px; color: #722ed1;"><span>3. Window Search Tree</span><span>61.2</span></div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px; color: #eb2f96;"><span>4. Post-filter</span><span>56.3</span></div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px; color: #8c8c8c;"><span>5. Pre-filter</span><span>46.1</span></div>
        </div>
        """
        st.markdown(score_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="custom-card"><div class="card-title">资源-性能权衡 <span class="info-icon">ⓘ</span></div>', unsafe_allow_html=True)
        fig4 = go.Figure()
        mem = [1.8, 3.2, 5.0, 6.7, 8.7]
        qps = [186, 462, 712, 956, 1284]
        names = ["Pre-filter", "Post-filter", "Window Search Tree", "SIEVE", "Optimized HNSW"]
        colors = ["#bfbfbf", "#91d5ff", "#722ed1", "#52c41a", "#1890ff"]
        sizes = [15, 20, 25, 30, 35]
        
        for i in range(len(names)):
            fig4.add_trace(go.Scatter(x=[mem[i]], y=[qps[i]], mode='markers+text', name=names[i], text=[names[i]], textposition="top center", marker=dict(color=colors[i], size=sizes[i], line=dict(width=1, color="#fff"))))
            
        fig4.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor="#fff", showlegend=False)
        fig4.update_xaxes(title="内存占用 (GB)", showgrid=True, gridcolor="#f0f0f0", range=[0, 10])
        fig4.update_yaxes(title="QPS (↑)", showgrid=True, gridcolor="#f0f0f0", range=[0, 1500])
        st.plotly_chart(fig4, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col5:
        st.markdown('<div class="custom-card"><div class="card-title">构建时间对比 <span class="info-icon">ⓘ</span></div>', unsafe_allow_html=True)
        fig5 = go.Figure()
        methods = ["Pre-filter", "Post-filter", "Window<br>Search Tree", "SIEVE", "Optimized<br>HNSW"]
        times = [28.6, 36.5, 54.2, 61.8, 72.4]
        colors = ["#bfbfbf", "#91d5ff", "#722ed1", "#52c41a", "#1890ff"]
        
        fig5.add_trace(go.Bar(x=methods, y=times, marker_color=colors, text=times, textposition='auto'))
        
        fig5.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor="#fff", showlegend=False)
        fig5.update_yaxes(title="构建时间 (分钟)", showgrid=True, gridcolor="#f0f0f0")
        st.plotly_chart(fig5, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    
    st.markdown('<div class="custom-card"><div class="card-title">算法对比明细表 <span class="info-icon">ⓘ</span></div>', unsafe_allow_html=True)
    table_html = """
    <table style="width: 100%; border-collapse: collapse; font-size: 13px; text-align: center; margin-top: 8px;">
        <tr style="background: #fafafa; color: #595959;">
            <th style="padding: 12px; border-bottom: 1px solid #f0f0f0;">方法</th>
            <th style="padding: 12px; border-bottom: 1px solid #f0f0f0;">过滤支持</th>
            <th style="padding: 12px; border-bottom: 1px solid #f0f0f0;">选择率</th>
            <th style="padding: 12px; border-bottom: 1px solid #f0f0f0;">Recall@10</th>
            <th style="padding: 12px; border-bottom: 1px solid #f0f0f0;">QPS</th>
            <th style="padding: 12px; border-bottom: 1px solid #f0f0f0;">P95 延迟</th>
            <th style="padding: 12px; border-bottom: 1px solid #f0f0f0;">内存</th>
            <th style="padding: 12px; border-bottom: 1px solid #f0f0f0;">构建时间</th>
            <th style="padding: 12px; border-bottom: 1px solid #f0f0f0;">推荐场景</th>
            <th style="padding: 12px; border-bottom: 1px solid #f0f0f0;">状态</th>
        </tr>
        <tr>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">Pre-filter</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #595959;">支持</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #595959;">1% ~ 100%</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">0.952</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">186</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">142.6ms</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">1.82GB</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">28.6 min</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #595959;">极低选择率，内存敏感场景</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0;"><span style="color: #52c41a; background: #f6ffed; border: 1px solid #b7eb8f; padding: 2px 6px; border-radius: 4px;">可用</span></td>
        </tr>
        <tr>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">Post-filter</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #595959;">支持</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #595959;">1% ~ 100%</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">0.965</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">462</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">78.4ms</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">3.21GB</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">36.5 min</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #595959;">通用场景，精度优先</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0;"><span style="color: #52c41a; background: #f6ffed; border: 1px solid #b7eb8f; padding: 2px 6px; border-radius: 4px;">可用</span></td>
        </tr>
        <tr>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">Window Search Tree</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #595959;">支持</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #595959;">1% ~ 100%</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">0.969</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">712</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">52.1ms</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">5.04GB</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">54.2 min</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #595959;">中高选择率，性能均衡</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0;"><span style="color: #52c41a; background: #f6ffed; border: 1px solid #b7eb8f; padding: 2px 6px; border-radius: 4px;">可用</span></td>
        </tr>
        <tr>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">SIEVE</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #595959;">支持</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #595959;">1% ~ 100%</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">0.971</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">956</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">31.6ms</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">6.73GB</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">61.8 min</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #595959;">低选择率，高性能需求</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0;"><span style="color: #1890ff; background: #e6f7ff; border: 1px solid #91d5ff; padding: 2px 6px; border-radius: 4px;">推荐</span></td>
        </tr>
        <tr>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626; font-weight: 600;">Optimized HNSW</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #595959; font-weight: 600;">支持</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #595959; font-weight: 600;">1% ~ 100%</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #1890ff; font-weight: 600;">0.973</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #1890ff; font-weight: 600;">1284</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #1890ff; font-weight: 600;">23.8ms</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626; font-weight: 600;">8.76GB</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626; font-weight: 600;">72.4 min</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #595959; font-weight: 600;">高性能通用场景（推荐）</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0;"><span style="color: #fff; background: #1890ff; border: 1px solid #1890ff; padding: 2px 6px; border-radius: 4px;">最佳</span></td>
        </tr>
    </table>
    """
    table_html = "".join([line.strip() for line in table_html.split("\n")])
    st.markdown(table_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# 页面 6: 参数敏感性分析 (Parameter Sensitivity)
# =============================================================================
def render_param_sens():
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: flex-end; padding-bottom: 16px; margin-bottom: 16px;">
        <div>
            <div style="font-size: 13px; color: #8c8c8c; margin-bottom: 6px;">首页 / 实验分析 / <span style="color: #262626;">参数敏感性分析</span></div>
            <div style="font-size: 22px; font-weight: 600; color: #262626; margin-bottom: 4px;">参数敏感性分析 <span class="info-icon">ⓘ</span></div>
            <div style="font-size: 13px; color: #8c8c8c;">分析 M、efConstruction、efSearch、过滤选择率、索引集合预取对 Recall、延迟、内存的影响</div>
        </div>
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="display: flex; align-items: center; gap: 6px; font-size: 13px; color: #595959;">
                <span>数据集</span>
                <select style="border: 1px solid #d9d9d9; border-radius: 4px; padding: 4px 8px; background: #fff; color: #595959; outline: none;"><option>Deep1B</option></select>
            </div>
            <div style="display: flex; align-items: center; gap: 6px; font-size: 13px; color: #595959;">
                <span>算法</span>
                <select style="border: 1px solid #d9d9d9; border-radius: 4px; padding: 4px 8px; background: #fff; color: #595959; outline: none;"><option>HNSW</option></select>
            </div>
            <div style="display: flex; align-items: center; gap: 6px; font-size: 13px; color: #595959;">
                <span>实验轮次</span>
                <select style="border: 1px solid #d9d9d9; border-radius: 4px; padding: 4px 8px; background: #fff; color: #595959; outline: none;"><option>Round 1</option></select>
            </div>
            <div style="display: flex; align-items: center; gap: 6px; font-size: 13px; color: #595959;">
                <span>批次/日期</span>
                <div style="border: 1px solid #d9d9d9; border-radius: 4px; padding: 4px 8px; background: #fff; display: flex; align-items: center; gap: 4px; color: #595959;">2025-05-20 📅</div>
            </div>
            <div style="display: flex; gap: 8px; margin-left: 8px;">
                <button style="border: 1px solid #d9d9d9; background: #fff; color: #1890ff; padding: 4px 16px; border-radius: 4px; cursor: pointer; font-size: 13px;">↻ 刷新</button>
                <button style="border: none; background: #1890ff; color: #fff; padding: 4px 16px; border-radius: 4px; cursor: pointer; font-size: 13px;">⬇ 导出</button>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    kpi_html = """
    <div style="background: #fff; border: 1px solid #d9d9d9; border-radius: 8px; display: flex; align-items: center; padding: 20px 0; margin-bottom: 24px;">
        <div style="flex: 1; padding: 0 24px; border-right: 1px solid #f0f0f0;">
            <div style="font-size: 14px; color: #595959; margin-bottom: 8px;">参数组合</div>
            <div style="font-size: 26px; font-weight: 500; color: #262626; margin-bottom: 4px;">216 <span style="color:#1890ff; font-size:20px; float:right;">🎛</span></div>
            <div style="font-size: 13px; color: #8c8c8c;"><span style="color: #1890ff;">↑ 12.5%</span> 较上次</div>
        </div>
        <div style="flex: 1; padding: 0 24px; border-right: 1px solid #f0f0f0;">
            <div style="font-size: 14px; color: #595959; margin-bottom: 8px;">最优 M</div>
            <div style="font-size: 26px; font-weight: 500; color: #262626; margin-bottom: 4px;">16 <span style="color:#1890ff; font-size:20px; float:right;">Ⓜ️</span></div>
            <div style="font-size: 13px; color: #8c8c8c;"><span style="color: #bfbfbf;">--</span> 较上次</div>
        </div>
        <div style="flex: 1; padding: 0 24px; border-right: 1px solid #f0f0f0;">
            <div style="font-size: 14px; color: #595959; margin-bottom: 8px;">最优 efSearch</div>
            <div style="font-size: 26px; font-weight: 500; color: #262626; margin-bottom: 4px;">100 <span style="color:#1890ff; font-size:20px; float:right;">🔍</span></div>
            <div style="font-size: 13px; color: #8c8c8c;"><span style="color: #1890ff;">↑ 8.7%</span> 较上次</div>
        </div>
        <div style="flex: 1; padding: 0 24px; border-right: 1px solid #f0f0f0;">
            <div style="font-size: 14px; color: #595959; margin-bottom: 8px;">Recall 提升</div>
            <div style="font-size: 26px; font-weight: 500; color: #262626; margin-bottom: 4px;">+4.6% <span style="color:#1890ff; font-size:20px; float:right;">📈</span></div>
            <div style="font-size: 13px; color: #8c8c8c;"><span style="color: #1890ff;">↑ 1.2%</span> 较上次</div>
        </div>
        <div style="flex: 1; padding: 0 24px; border-right: 1px solid #f0f0f0;">
            <div style="font-size: 14px; color: #595959; margin-bottom: 8px;">延迟拐点</div>
            <div style="font-size: 26px; font-weight: 500; color: #262626; margin-bottom: 4px;">ef=200 <span style="color:#1890ff; font-size:20px; float:right;">⏱</span></div>
            <div style="font-size: 13px; color: #8c8c8c;"><span style="color: #52c41a;">↓ 18.6%</span> 拐点处斜率</div>
        </div>
        <div style="flex: 1; padding: 0 24px;">
            <div style="font-size: 14px; color: #595959; margin-bottom: 8px;">内存抖率</div>
            <div style="font-size: 26px; font-weight: 500; color: #262626; margin-bottom: 4px;">0.38<span style="font-size:16px;">GB/M</span> <span style="color:#1890ff; font-size:20px; float:right;">🖧</span></div>
            <div style="font-size: 13px; color: #8c8c8c;"><span style="color: #1890ff;">↑ 3.1%</span> 较上次</div>
        </div>
    </div>
    """
    kpi_html = "".join([line.strip() for line in kpi_html.split("\n")])
    st.markdown(kpi_html, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1.2, 1, 1])

    with col1:
        st.markdown('<div class="custom-card"><div class="card-title">M x efSearch Recall 热力图 <span class="info-icon">ⓘ</span></div>', unsafe_allow_html=True)
        z = [[0.786, 0.843, 0.881, 0.905, 0.923, 0.934],
             [0.841, 0.904, 0.942, 0.968, 0.979, 0.984],
             [0.861, 0.922, 0.958, 0.977, 0.985, 0.988],
             [0.868, 0.925, 0.960, 0.978, 0.987, 0.989]]
        x = ['20', '50', '100', '200', '400', '800']
        y = ['8', '16', '32', '64']
        
        fig1 = go.Figure(data=go.Heatmap(
            z=z, x=x, y=y,
            colorscale=[[0, '#e6f7ff'], [0.5, '#69b1ff'], [1, '#0050b3']],
            text=[[str(val) for val in row] for row in z],
            texttemplate="%{text}", textfont={"size":11}
        ))
        fig1.update_layout(height=240, margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor="#fff", xaxis_title="efSearch", yaxis_title="M")
        st.plotly_chart(fig1, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="custom-card"><div class="card-title">efSearch 对延迟影响</div>', unsafe_allow_html=True)
        fig2 = go.Figure()
        ef = [20, 50, 100, 200, 400, 800]
        p50 = [0, 2, 5, 20, 35, 60]
        p95 = [22, 36, 58, 102, 170, 245]
        
        fig2.add_trace(go.Scatter(x=ef, y=p50, name="P50 延迟", mode="lines+markers", line=dict(color="#69b1ff", width=2, dash="dash"), marker=dict(symbol="circle", size=6)))
        fig2.add_trace(go.Scatter(x=ef, y=p95, name="P95 延迟", mode="lines+markers+text", line=dict(color="#1890ff", width=2), marker=dict(symbol="circle", size=6), text=p95, textposition="top left", textfont=dict(size=10)))
        
        fig2.update_layout(height=240, margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor="#fff", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=10)), xaxis_type="log")
        fig2.update_xaxes(title="efSearch", showgrid=True, gridcolor="#f0f0f0", tickvals=ef, ticktext=ef)
        fig2.update_yaxes(title="延迟 (ms)", showgrid=True, gridcolor="#f0f0f0", range=[0, 300])
        st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="custom-card"><div class="card-title">M 对内存占用影响</div>', unsafe_allow_html=True)
        fig3 = go.Figure()
        m_vals = [8, 16, 32, 64, 128]
        mem = [3.1, 6.2, 11.8, 22.9, 44.6]
        
        fig3.add_trace(go.Scatter(x=m_vals, y=mem, name="实测内存 (GB)", mode="lines+markers+text", line=dict(color="#1890ff", width=2), marker=dict(symbol="circle", size=6), text=mem, textposition="top left", textfont=dict(size=10)))
        
        fig3.update_layout(height=240, margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor="#fff", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=10)), xaxis_type="log")
        fig3.update_xaxes(title="M", showgrid=True, gridcolor="#f0f0f0", tickvals=m_vals, ticktext=m_vals)
        fig3.update_yaxes(title="Memory (GB)", showgrid=True, gridcolor="#f0f0f0", range=[0, 50])
        st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    
    col4, col5 = st.columns([1, 1])
    
    with col4:
        st.markdown('<div class="custom-card"><div class="card-title">选择率敏感性 <span class="info-icon">ⓘ</span></div>', unsafe_allow_html=True)
        fig4 = make_subplots(specs=[[{"secondary_y": True}]])
        sel_str = ["1%", "2%", "5%", "10%", "25%", "50%", "100%"]
        
        fig4.add_trace(go.Scatter(x=sel_str, y=[92.1, 88.7, 78.6, 64.3, 41.2, 23.7, 11.8], name="QPS (K)", mode="lines+markers", line=dict(color="#1890ff", width=2)), secondary_y=False)
        fig4.add_trace(go.Scatter(x=sel_str, y=[100, 95, 85, 70, 45, 25, 12], name="QPS (K) 优化后", mode="lines+markers", line=dict(color="#1890ff", width=2, dash="dash")), secondary_y=False)
        fig4.add_trace(go.Scatter(x=sel_str, y=[71.4, 67.8, 59.2, 48.1, 29.7, 16.0, 7.6], name="延迟 SIEVE 过滤 (ms)", mode="lines+markers", line=dict(color="#52c41a", width=2)), secondary_y=True)
        fig4.add_trace(go.Scatter(x=sel_str, y=[60, 55, 45, 35, 20, 10, 5], name="延迟 优化后 (ms)", mode="lines+markers", line=dict(color="#52c41a", width=2, dash="dash")), secondary_y=True)
        
        fig4.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor="#fff", legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5, font=dict(size=10)))
        fig4.update_xaxes(title="选择率", showgrid=True, gridcolor="#f0f0f0")
        fig4.update_yaxes(title="QPS (K)", range=[0, 120], showgrid=True, gridcolor="#f0f0f0", secondary_y=False)
        fig4.update_yaxes(title="延迟 (ms)", range=[0, 100], showgrid=False, secondary_y=True)
        st.plotly_chart(fig4, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col5:
        st.markdown('<div class="custom-card"><div class="card-title">Pareto 参数候选</div>', unsafe_allow_html=True)
        fig5 = go.Figure()
        recalls = np.linspace(0.82, 1.0, 20)
        delays = 250 * np.exp(-15 * (recalls - 0.82))
        
        fig5.add_trace(go.Scatter(x=recalls, y=delays, mode="lines+markers", name="Pareto Front", line=dict(color="#1890ff", width=2), marker=dict(size=5)))
        fig5.add_trace(go.Scatter(x=[0.942], y=[48.1], mode="markers", name="推荐点", marker=dict(symbol="star", size=12, color="#1890ff", line=dict(width=1, color="#0050b3"))))
        
        fig5.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor="#fff", showlegend=False)
        fig5.update_xaxes(title="Recall@10", showgrid=True, gridcolor="#f0f0f0")
        fig5.update_yaxes(title="P95 延迟 (ms)", range=[0, 250], showgrid=True, gridcolor="#f0f0f0")
        
        # Overlay a recommendation card inside the chart area
        fig5.add_annotation(
            x=0.98, y=180,
            text="<b>推荐组合 (Pareto 最优)</b><br><br>M=16, efConstruction=200,<br>efSearch=100<br><br>Recall@10: 0.942<br>P95 延迟: 48.1ms<br>内存: 6.2GB",
            showarrow=False, align="left", bgcolor="#fff", bordercolor="#e8e8e8", borderwidth=1, borderpad=8, font=dict(size=11, color="#595959")
        )
        st.plotly_chart(fig5, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    
    st.markdown('<div class="custom-card"><div class="card-title">参数实验明细表 <span class="info-icon">ⓘ</span></div>', unsafe_allow_html=True)
    table_html = """
    <table style="width: 100%; border-collapse: collapse; font-size: 13px; text-align: center; margin-top: 8px;">
        <tr style="background: #fafafa; color: #595959;">
            <th style="padding: 12px; border-bottom: 1px solid #f0f0f0;">组合ID</th>
            <th style="padding: 12px; border-bottom: 1px solid #f0f0f0;">M</th>
            <th style="padding: 12px; border-bottom: 1px solid #f0f0f0;">efC</th>
            <th style="padding: 12px; border-bottom: 1px solid #f0f0f0;">efS</th>
            <th style="padding: 12px; border-bottom: 1px solid #f0f0f0;">选择率</th>
            <th style="padding: 12px; border-bottom: 1px solid #f0f0f0;">Recall@10</th>
            <th style="padding: 12px; border-bottom: 1px solid #f0f0f0;">QPS</th>
            <th style="padding: 12px; border-bottom: 1px solid #f0f0f0;">P95延迟(ms)</th>
            <th style="padding: 12px; border-bottom: 1px solid #f0f0f0;">内存(GB)</th>
            <th style="padding: 12px; border-bottom: 1px solid #f0f0f0;">结论</th>
            <th style="padding: 12px; border-bottom: 1px solid #f0f0f0;">状态</th>
        </tr>
        <tr>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #595959;">P-001</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">8</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">100</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">50</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #595959;">10%</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">0.904</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">64.3K</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">36</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">3.1</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #595959;">延迟低，召回中等，内存最小</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0;"><span style="color: #52c41a; background: #f6ffed; border: 1px solid #b7eb8f; padding: 2px 6px; border-radius: 4px;">完成</span></td>
        </tr>
        <tr style="background-color: #f0f5ff;">
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #1890ff; font-weight: 500;">P-002</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #1890ff; font-weight: 500;">16</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #1890ff; font-weight: 500;">200</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #1890ff; font-weight: 500;">100</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #1890ff; font-weight: 500;">10%</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #1890ff; font-weight: 500;">0.942</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #1890ff; font-weight: 500;">48.1K</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #1890ff; font-weight: 500;">58</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #1890ff; font-weight: 500;">6.2</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #1890ff; font-weight: 500;">推荐组合，性能均衡最佳</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0;"><span style="color: #52c41a; background: #f6ffed; border: 1px solid #b7eb8f; padding: 2px 6px; border-radius: 4px;">完成</span></td>
        </tr>
        <tr>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #595959;">P-003</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">16</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">200</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">200</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #595959;">10%</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">0.968</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">35.0K</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">102</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">6.3</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #595959;">召回更高，延迟上升</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0;"><span style="color: #52c41a; background: #f6ffed; border: 1px solid #b7eb8f; padding: 2px 6px; border-radius: 4px;">完成</span></td>
        </tr>
        <tr>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #595959;">P-004</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">32</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">200</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">100</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #595959;">10%</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">0.958</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">42.5K</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">61</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">11.8</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #595959;">召回略低于 P-003，内存增大</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0;"><span style="color: #52c41a; background: #f6ffed; border: 1px solid #b7eb8f; padding: 2px 6px; border-radius: 4px;">完成</span></td>
        </tr>
    </table>
    """
    table_html = "".join([line.strip() for line in table_html.split("\n")])
    st.markdown(table_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# 页面 7: HNSW 索引构建 (HNSW Index Construction)
# =============================================================================
def render_build_index():
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: flex-end; padding-bottom: 16px; margin-bottom: 16px;">
        <div>
            <div style="font-size: 13px; color: #8c8c8c; margin-bottom: 6px;">首页 / 实验分析 / <span style="color: #262626;">HNSW 索引构建</span></div>
            <div style="font-size: 22px; font-weight: 600; color: #262626; margin-bottom: 4px;">HNSW 索引构建</div>
            <div style="font-size: 13px; color: #8c8c8c;">监控 HNSW 图索引构建、参数组合、层级结构、内存占用与构建日志</div>
        </div>
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="display: flex; align-items: center; gap: 6px; font-size: 13px; color: #595959;">
                <span>数据集</span>
                <select style="border: 1px solid #d9d9d9; border-radius: 4px; padding: 4px 8px; background: #fff; color: #595959; outline: none;"><option>Deep1B</option></select>
            </div>
            <div style="display: flex; align-items: center; gap: 6px; font-size: 13px; color: #595959;">
                <span>算法</span>
                <select style="border: 1px solid #d9d9d9; border-radius: 4px; padding: 4px 8px; background: #fff; color: #595959; outline: none;"><option>HNSW</option></select>
            </div>
            <div style="display: flex; align-items: center; gap: 6px; font-size: 13px; color: #595959;">
                <span>实验轮次</span>
                <select style="border: 1px solid #d9d9d9; border-radius: 4px; padding: 4px 8px; background: #fff; color: #595959; outline: none;"><option>Round 1</option></select>
            </div>
            <div style="display: flex; align-items: center; gap: 6px; font-size: 13px; color: #595959;">
                <span>批次/日期</span>
                <div style="border: 1px solid #d9d9d9; border-radius: 4px; padding: 4px 8px; background: #fff; display: flex; align-items: center; gap: 4px; color: #595959;">2025-05-20 📅</div>
            </div>
            <div style="display: flex; gap: 8px; margin-left: 8px;">
                <button style="border: 1px solid #d9d9d9; background: #fff; color: #1890ff; padding: 4px 16px; border-radius: 4px; cursor: pointer; font-size: 13px;">↻ 刷新</button>
                <button style="border: none; background: #1890ff; color: #fff; padding: 4px 16px; border-radius: 4px; cursor: pointer; font-size: 13px;">⬇ 导出</button>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    kpi_html = """
    <div style="background: #fff; border: 1px solid #d9d9d9; border-radius: 8px; display: flex; align-items: center; padding: 20px 0; margin-bottom: 24px;">
        <div style="flex: 1; padding: 0 24px; border-right: 1px solid #f0f0f0;">
            <div style="font-size: 14px; color: #595959; margin-bottom: 8px;">构建任务</div>
            <div style="font-size: 26px; font-weight: 500; color: #262626; margin-bottom: 4px;">12 <span style="color:#1890ff; font-size:20px; float:right;">📋</span></div>
            <div style="font-size: 13px; color: #8c8c8c;"><span style="color: #bfbfbf;">--</span></div>
        </div>
        <div style="flex: 1; padding: 0 24px; border-right: 1px solid #f0f0f0;">
            <div style="font-size: 14px; color: #595959; margin-bottom: 8px;">当前 M</div>
            <div style="font-size: 26px; font-weight: 500; color: #262626; margin-bottom: 4px;">16 <span style="color:#1890ff; font-size:20px; float:right;">Ⓜ️</span></div>
            <div style="font-size: 13px; color: #8c8c8c;"><span style="color: #bfbfbf;">--</span></div>
        </div>
        <div style="flex: 1; padding: 0 24px; border-right: 1px solid #f0f0f0;">
            <div style="font-size: 14px; color: #595959; margin-bottom: 8px;">efConstruction</div>
            <div style="font-size: 26px; font-weight: 500; color: #262626; margin-bottom: 4px;">200 <span style="color:#1890ff; font-size:20px; float:right;">⚙️</span></div>
            <div style="font-size: 13px; color: #8c8c8c;"><span style="color: #bfbfbf;">--</span></div>
        </div>
        <div style="flex: 1; padding: 0 24px; border-right: 1px solid #f0f0f0;">
            <div style="font-size: 14px; color: #595959; margin-bottom: 8px;">插入吞吐 (kips)</div>
            <div style="font-size: 26px; font-weight: 500; color: #262626; margin-bottom: 4px;">42.6 <span style="color:#1890ff; font-size:20px; float:right;">🚀</span></div>
            <div style="font-size: 13px; color: #8c8c8c;"><span style="color: #bfbfbf;">--</span></div>
        </div>
        <div style="flex: 1; padding: 0 24px; border-right: 1px solid #f0f0f0;">
            <div style="font-size: 14px; color: #595959; margin-bottom: 8px;">峰值内存 (GB)</div>
            <div style="font-size: 26px; font-weight: 500; color: #262626; margin-bottom: 4px;">6.42 <span style="color:#1890ff; font-size:20px; float:right;">💾</span></div>
            <div style="font-size: 13px; color: #8c8c8c;"><span style="color: #bfbfbf;">--</span></div>
        </div>
        <div style="flex: 1; padding: 0 24px;">
            <div style="font-size: 14px; color: #595959; margin-bottom: 8px;">索引文件</div>
            <div style="font-size: 26px; font-weight: 500; color: #262626; margin-bottom: 4px;">5.88<span style="font-size: 16px;">GB</span> <span style="color:#1890ff; font-size:20px; float:right;">📁</span></div>
            <div style="font-size: 13px; color: #8c8c8c;"><span style="color: #bfbfbf;">--</span></div>
        </div>
    </div>
    """
    kpi_html = "".join([line.strip() for line in kpi_html.split("\n")])
    st.markdown(kpi_html, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown('<div class="custom-card"><div class="card-title">构建进度与阶段</div>', unsafe_allow_html=True)
        progress_html = """
        <div style="margin-top: 16px; font-size: 13px; color: #595959;">
            <div style="display: flex; align-items: center; margin-bottom: 16px;">
                <div style="width: 24px; height: 24px; border-radius: 50%; background: #52c41a; color: white; display: flex; align-items: center; justify-content: center; font-size: 12px; margin-right: 12px;">✓</div>
                <div style="flex: 1;">
                    <div style="font-weight: 500; color: #262626;">加载向量</div>
                    <div style="font-size: 11px; color: #8c8c8c;">读取数据集并初始化构建环境</div>
                </div>
                <div style="width: 80px; display: flex; align-items: center; gap: 8px;">
                    <div style="flex: 1; height: 6px; background: #e8e8e8; border-radius: 3px;"><div style="width: 100%; height: 100%; background: #1890ff; border-radius: 3px;"></div></div>
                    <div style="font-size: 12px; width: 30px; text-align: right;">100%</div>
                </div>
            </div>
            
            <div style="display: flex; align-items: center; margin-bottom: 16px;">
                <div style="width: 24px; height: 24px; border-radius: 50%; background: #52c41a; color: white; display: flex; align-items: center; justify-content: center; font-size: 12px; margin-right: 12px;">✓</div>
                <div style="flex: 1;">
                    <div style="font-weight: 500; color: #262626;">初始化入口点</div>
                    <div style="font-size: 11px; color: #8c8c8c;">选择初始入口集合</div>
                </div>
                <div style="width: 80px; display: flex; align-items: center; gap: 8px;">
                    <div style="flex: 1; height: 6px; background: #e8e8e8; border-radius: 3px;"><div style="width: 100%; height: 100%; background: #1890ff; border-radius: 3px;"></div></div>
                    <div style="font-size: 12px; width: 30px; text-align: right;">100%</div>
                </div>
            </div>
            
            <div style="display: flex; align-items: center; margin-bottom: 16px;">
                <div style="width: 24px; height: 24px; border-radius: 50%; background: #1890ff; color: white; display: flex; align-items: center; justify-content: center; font-size: 12px; margin-right: 12px;">3</div>
                <div style="flex: 1;">
                    <div style="font-weight: 500; color: #1890ff;">增量插入</div>
                    <div style="font-size: 11px; color: #8c8c8c;">逐步插入向量并建立图连接</div>
                </div>
                <div style="width: 80px; display: flex; align-items: center; gap: 8px;">
                    <div style="flex: 1; height: 6px; background: #e8e8e8; border-radius: 3px;"><div style="width: 78%; height: 100%; background: #1890ff; border-radius: 3px;"></div></div>
                    <div style="font-size: 12px; width: 30px; text-align: right; color: #1890ff;">78%</div>
                </div>
            </div>
            
            <div style="display: flex; align-items: center; margin-bottom: 16px;">
                <div style="width: 24px; height: 24px; border-radius: 50%; background: #1890ff; color: white; display: flex; align-items: center; justify-content: center; font-size: 12px; margin-right: 12px;">4</div>
                <div style="flex: 1;">
                    <div style="font-weight: 500; color: #1890ff;">邻居选择启发式</div>
                    <div style="font-size: 11px; color: #8c8c8c;">应用启发式裁剪策略优化连接</div>
                </div>
                <div style="width: 80px; display: flex; align-items: center; gap: 8px;">
                    <div style="flex: 1; height: 6px; background: #e8e8e8; border-radius: 3px;"><div style="width: 78%; height: 100%; background: #1890ff; border-radius: 3px;"></div></div>
                    <div style="font-size: 12px; width: 30px; text-align: right; color: #1890ff;">78%</div>
                </div>
            </div>
            
            <div style="display: flex; align-items: center;">
                <div style="width: 24px; height: 24px; border-radius: 50%; border: 1px solid #d9d9d9; background: #fff; color: #bfbfbf; display: flex; align-items: center; justify-content: center; font-size: 12px; margin-right: 12px;">5</div>
                <div style="flex: 1;">
                    <div style="font-weight: 500; color: #bfbfbf;">序列化索引</div>
                    <div style="font-size: 11px; color: #bfbfbf;">持久化索引文件与元数据</div>
                </div>
                <div style="width: 80px; display: flex; align-items: center; justify-content: flex-end;">
                    <div style="font-size: 12px; color: #bfbfbf;">等待中</div>
                </div>
            </div>
        </div>
        """
        progress_html = "".join([line.strip() for line in progress_html.split("\n")])
        st.markdown(progress_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="custom-card"><div class="card-title">HNSW 层级图结构 <span class="info-icon">ⓘ</span></div>', unsafe_allow_html=True)
        graph_html = """
        <div style="height: 220px; display: flex; flex-direction: column; justify-content: center; align-items: center; position: relative;">
            <!-- Placeholder for actual network graph, simplified visual representation -->
            <div style="font-size: 12px; color: #8c8c8c; position: absolute; left: 0; top: 10px;">Layer 3</div>
            <div style="font-size: 12px; color: #1890ff; position: absolute; left: 45px; top: 10px; background: #e6f7ff; padding: 2px 6px; border-radius: 10px;">3</div>
            
            <div style="font-size: 12px; color: #8c8c8c; position: absolute; left: 0; top: 60px;">Layer 2</div>
            <div style="font-size: 12px; color: #1890ff; position: absolute; left: 45px; top: 60px; background: #e6f7ff; padding: 2px 6px; border-radius: 10px;">7</div>
            
            <div style="font-size: 12px; color: #8c8c8c; position: absolute; left: 0; top: 110px;">Layer 1</div>
            <div style="font-size: 12px; color: #1890ff; position: absolute; left: 45px; top: 110px; background: #e6f7ff; padding: 2px 6px; border-radius: 10px;">16</div>
            
            <div style="font-size: 12px; color: #8c8c8c; position: absolute; left: 0; top: 160px;">Layer 0</div>
            <div style="font-size: 12px; color: #1890ff; position: absolute; left: 45px; top: 160px; background: #e6f7ff; padding: 2px 6px; border-radius: 10px;">32</div>
            
            <svg width="200" height="180">
                <!-- Layer 3 to Layer 2 -->
                <line x1="100" y1="20" x2="80" y2="70" stroke="#1890ff" stroke-width="1.5" />
                <line x1="100" y1="20" x2="120" y2="70" stroke="#1890ff" stroke-width="1.5" />
                
                <!-- Layer 2 to Layer 1 -->
                <line x1="80" y1="70" x2="60" y2="120" stroke="#1890ff" stroke-width="1.5" />
                <line x1="80" y1="70" x2="100" y2="120" stroke="#1890ff" stroke-width="1.5" />
                <line x1="120" y1="70" x2="100" y2="120" stroke="#1890ff" stroke-width="1.5" />
                <line x1="120" y1="70" x2="140" y2="120" stroke="#1890ff" stroke-width="1.5" />
                
                <!-- Layer 1 to Layer 0 -->
                <line x1="60" y1="120" x2="40" y2="170" stroke="#bfbfbf" stroke-width="1" stroke-dasharray="2,2" />
                <line x1="60" y1="120" x2="80" y2="170" stroke="#bfbfbf" stroke-width="1" stroke-dasharray="2,2" />
                <line x1="100" y1="120" x2="80" y2="170" stroke="#bfbfbf" stroke-width="1" stroke-dasharray="2,2" />
                <line x1="100" y1="120" x2="120" y2="170" stroke="#bfbfbf" stroke-width="1" stroke-dasharray="2,2" />
                <line x1="140" y1="120" x2="120" y2="170" stroke="#bfbfbf" stroke-width="1" stroke-dasharray="2,2" />
                <line x1="140" y1="120" x2="160" y2="170" stroke="#bfbfbf" stroke-width="1" stroke-dasharray="2,2" />
                
                <!-- Nodes -->
                <circle cx="100" cy="20" r="6" fill="#fff" stroke="#1890ff" stroke-width="2" />
                
                <circle cx="80" cy="70" r="5" fill="#fff" stroke="#1890ff" stroke-width="2" />
                <circle cx="120" cy="70" r="5" fill="#fff" stroke="#1890ff" stroke-width="2" />
                <line x1="80" y1="70" x2="120" y2="70" stroke="#1890ff" stroke-width="1" stroke-dasharray="2,2" />
                
                <circle cx="60" cy="120" r="5" fill="#fff" stroke="#1890ff" stroke-width="2" />
                <circle cx="100" cy="120" r="5" fill="#fff" stroke="#1890ff" stroke-width="2" />
                <circle cx="140" cy="120" r="5" fill="#fff" stroke="#1890ff" stroke-width="2" />
                <line x1="60" y1="120" x2="100" y2="120" stroke="#1890ff" stroke-width="1" stroke-dasharray="2,2" />
                <line x1="100" y1="120" x2="140" y2="120" stroke="#1890ff" stroke-width="1" stroke-dasharray="2,2" />
                
                <circle cx="40" cy="170" r="4" fill="#e8e8e8" />
                <circle cx="80" cy="170" r="4" fill="#e8e8e8" />
                <circle cx="120" cy="170" r="4" fill="#e8e8e8" />
                <circle cx="160" cy="170" r="4" fill="#e8e8e8" />
            </svg>
            
            <div style="font-size: 11px; color: #595959; margin-top: 8px;"><span style="color:#1890ff;">●</span> 入口点</div>
        </div>
        """
        graph_html = "".join([line.strip() for line in graph_html.split("\n")])
        st.markdown(graph_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="custom-card"><div class="card-title">参数组合队列 <span class="info-icon">ⓘ</span></div>', unsafe_allow_html=True)
        queue_html = """
        <table style="width: 100%; border-collapse: collapse; font-size: 12px; text-align: center; margin-top: 16px;">
            <tr style="background: #fafafa; color: #595959;">
                <th style="padding: 8px 4px; border-bottom: 1px solid #f0f0f0;">M</th>
                <th style="padding: 8px 4px; border-bottom: 1px solid #f0f0f0;">efC</th>
                <th style="padding: 8px 4px; border-bottom: 1px solid #f0f0f0;">efS</th>
                <th style="padding: 8px 4px; border-bottom: 1px solid #f0f0f0;">Recall</th>
                <th style="padding: 8px 4px; border-bottom: 1px solid #f0f0f0;">状态</th>
                <th style="padding: 8px 4px; border-bottom: 1px solid #f0f0f0;">进度</th>
            </tr>
            <tr>
                <td style="padding: 10px 4px; border-bottom: 1px solid #f0f0f0;">8</td>
                <td style="padding: 10px 4px; border-bottom: 1px solid #f0f0f0;">100</td>
                <td style="padding: 10px 4px; border-bottom: 1px solid #f0f0f0;">50</td>
                <td style="padding: 10px 4px; border-bottom: 1px solid #f0f0f0;">0.912</td>
                <td style="padding: 10px 4px; border-bottom: 1px solid #f0f0f0;"><span style="color: #52c41a; background: #f6ffed; padding: 2px 6px; border-radius: 4px;">完成</span></td>
                <td style="padding: 10px 4px; border-bottom: 1px solid #f0f0f0;">
                    <div style="display:flex; align-items:center; gap:4px;"><div style="flex:1; height:4px; background:#e8e8e8; border-radius:2px;"><div style="width:100%; height:100%; background:#52c41a; border-radius:2px;"></div></div>100%</div>
                </td>
            </tr>
            <tr style="background: #f0f5ff;">
                <td style="padding: 10px 4px; border-bottom: 1px solid #f0f0f0; color: #1890ff; font-weight: 500;">16</td>
                <td style="padding: 10px 4px; border-bottom: 1px solid #f0f0f0; color: #1890ff; font-weight: 500;">200</td>
                <td style="padding: 10px 4px; border-bottom: 1px solid #f0f0f0; color: #1890ff; font-weight: 500;">100</td>
                <td style="padding: 10px 4px; border-bottom: 1px solid #f0f0f0; color: #1890ff; font-weight: 500;">0.947</td>
                <td style="padding: 10px 4px; border-bottom: 1px solid #f0f0f0;"><span style="color: #1890ff; background: #e6f7ff; border: 1px solid #91d5ff; padding: 2px 6px; border-radius: 4px;">运行中</span></td>
                <td style="padding: 10px 4px; border-bottom: 1px solid #f0f0f0;">
                    <div style="display:flex; align-items:center; gap:4px;"><div style="flex:1; height:4px; background:#e8e8e8; border-radius:2px;"><div style="width:78%; height:100%; background:#1890ff; border-radius:2px;"></div></div>78%</div>
                </td>
            </tr>
            <tr>
                <td style="padding: 10px 4px; border-bottom: 1px solid #f0f0f0;">32</td>
                <td style="padding: 10px 4px; border-bottom: 1px solid #f0f0f0;">200</td>
                <td style="padding: 10px 4px; border-bottom: 1px solid #f0f0f0;">200</td>
                <td style="padding: 10px 4px; border-bottom: 1px solid #f0f0f0;">0.961</td>
                <td style="padding: 10px 4px; border-bottom: 1px solid #f0f0f0;"><span style="color: #8c8c8c; background: #fafafa; padding: 2px 6px; border-radius: 4px;">排队</span></td>
                <td style="padding: 10px 4px; border-bottom: 1px solid #f0f0f0;">
                    <div style="display:flex; align-items:center; gap:4px;"><div style="flex:1; height:4px; background:#e8e8e8; border-radius:2px;"><div style="width:0%; height:100%; background:#bfbfbf; border-radius:2px;"></div></div>0%</div>
                </td>
            </tr>
            <tr>
                <td style="padding: 10px 4px; border-bottom: 1px solid #f0f0f0;">16</td>
                <td style="padding: 10px 4px; border-bottom: 1px solid #f0f0f0;">400</td>
                <td style="padding: 10px 4px; border-bottom: 1px solid #f0f0f0;">100</td>
                <td style="padding: 10px 4px; border-bottom: 1px solid #f0f0f0;">0.952</td>
                <td style="padding: 10px 4px; border-bottom: 1px solid #f0f0f0;"><span style="color: #8c8c8c; background: #fafafa; padding: 2px 6px; border-radius: 4px;">排队</span></td>
                <td style="padding: 10px 4px; border-bottom: 1px solid #f0f0f0;">
                    <div style="display:flex; align-items:center; gap:4px;"><div style="flex:1; height:4px; background:#e8e8e8; border-radius:2px;"><div style="width:0%; height:100%; background:#bfbfbf; border-radius:2px;"></div></div>0%</div>
                </td>
            </tr>
        </table>
        """
        queue_html = "".join([line.strip() for line in queue_html.split("\n")])
        st.markdown(queue_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    
    col4, col5, col6 = st.columns([1, 1, 1])
    
    with col4:
        st.markdown('<div class="custom-card"><div class="card-title">构建期内存变化</div>', unsafe_allow_html=True)
        fig_m = go.Figure()
        progress = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        mem = [0.5, 1.2, 2.5, 3.8, 4.6, 5.2, 5.8, 6.2, 6.42, 6.42, 6.42]
        
        fig_m.add_trace(go.Scatter(x=progress, y=mem, mode="lines+markers", name="内存使用", line=dict(color="#1890ff", width=2), fill='tozeroy', fillcolor='rgba(24,144,255,0.1)'))
        fig_m.add_trace(go.Scatter(x=[0, 100], y=[6.42, 6.42], mode="lines", name="峰值内存 6.42 GB", line=dict(color="#bfbfbf", width=1, dash="dash")))
        
        fig_m.update_layout(height=240, margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor="#fff", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=10)))
        fig_m.update_xaxes(title="插入比例 (%)", showgrid=True, gridcolor="#f0f0f0", range=[0, 100])
        fig_m.update_yaxes(title="GB", showgrid=True, gridcolor="#f0f0f0", range=[0, 10])
        
        fig_m.add_annotation(x=80, y=7, text="峰值 6.42 GB", showarrow=False, bgcolor="#fff", bordercolor="#d9d9d9", borderwidth=1, borderpad=4, font=dict(size=10, color="#595959"))
        
        st.plotly_chart(fig_m, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col5:
        st.markdown('<div class="custom-card"><div class="card-title">插入吞吐趋势</div>', unsafe_allow_html=True)
        fig_t = go.Figure()
        time_steps = ["11:50", "11:55", "12:00", "12:05", "12:10"]
        x_vals = np.linspace(0, 4, 15)
        throughput = 20 + 40 * np.exp(-((x_vals - 2.5) ** 2) / 2) + np.random.normal(0, 2, 15)
        throughput[throughput > 60] = 58.3
        
        fig_t.add_trace(go.Scatter(x=x_vals, y=throughput, mode="lines+markers", name="吞吐 (kips)", line=dict(color="#1890ff", width=2)))
        fig_t.add_trace(go.Scatter(x=[0, 4], y=[42.6, 42.6], mode="lines", name="均值 42.6 kips", line=dict(color="#bfbfbf", width=1, dash="dash")))
        
        fig_t.update_layout(height=240, margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor="#fff", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=10)))
        fig_t.update_xaxes(title="时间", showgrid=True, gridcolor="#f0f0f0", tickvals=[0, 1, 2, 3, 4], ticktext=time_steps)
        fig_t.update_yaxes(title="kips", showgrid=True, gridcolor="#f0f0f0", range=[0, 100])
        
        fig_t.add_annotation(x=3, y=65, text="峰值 58.3 kips", showarrow=False, bgcolor="#fff", bordercolor="#d9d9d9", borderwidth=1, borderpad=4, font=dict(size=10, color="#595959"))
        
        st.plotly_chart(fig_t, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col6:
        st.markdown('<div class="custom-card"><div class="card-title">构建日志摘录</div>', unsafe_allow_html=True)
        logs_html = """
        <div style="background: #fafafa; border: 1px solid #f0f0f0; border-radius: 4px; padding: 12px; height: 180px; overflow-y: auto; font-family: monospace; font-size: 11px; margin-bottom: 12px;">
            <div style="margin-bottom: 6px;"><span style="color: #8c8c8c;">[12:01:02]</span> <span style="color: #1890ff; background: #e6f7ff; padding: 0 4px; border-radius: 2px;">INFO</span> load base.npy: 1,000,000 vectors</div>
            <div style="margin-bottom: 6px;"><span style="color: #8c8c8c;">[12:04:11]</span> <span style="color: #1890ff; background: #e6f7ff; padding: 0 4px; border-radius: 2px;">INFO</span> insert 780k / 1M (78.0%)</div>
            <div style="margin-bottom: 6px;"><span style="color: #8c8c8c;">[12:05:28]</span> <span style="color: #722ed1; background: #f9f0ff; padding: 0 4px; border-radius: 2px;">DEBUG</span> neighbor heuristic enabled (efC = 200)</div>
            <div style="margin-bottom: 6px;"><span style="color: #8c8c8c;">[12:07:44]</span> <span style="color: #1890ff; background: #e6f7ff; padding: 0 4px; border-radius: 2px;">INFO</span> memory RSS 6.42 GB</div>
            <div style="margin-bottom: 6px;"><span style="color: #8c8c8c;">[12:07:45]</span> <span style="color: #1890ff; background: #e6f7ff; padding: 0 4px; border-radius: 2px;">INFO</span> current throughput 42.6 kips</div>
        </div>
        <div style="font-size: 12px; color: #595959; display: flex; align-items: center; gap: 12px;">
            日志级别：
            <span style="display: flex; align-items: center; gap: 4px;"><div style="width:8px; height:8px; border-radius:50%; background:#1890ff;"></div> INFO</span>
            <span style="display: flex; align-items: center; gap: 4px;"><div style="width:8px; height:8px; border-radius:50%; background:#722ed1;"></div> DEBUG</span>
            <span style="display: flex; align-items: center; gap: 4px;"><div style="width:8px; height:8px; border-radius:50%; background:#faad14;"></div> WARN</span>
            <span style="display: flex; align-items: center; gap: 4px;"><div style="width:8px; height:8px; border-radius:50%; background:#ff4d4f;"></div> ERROR</span>
        </div>
        """
        logs_html = "".join([line.strip() for line in logs_html.split("\n")])
        st.markdown(logs_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    
    st.markdown('<div class="custom-card"><div class="card-title">索引构建任务表</div>', unsafe_allow_html=True)
    table_html = """
    <table style="width: 100%; border-collapse: collapse; font-size: 13px; text-align: center; margin-top: 8px;">
        <tr style="background: #fafafa; color: #595959;">
            <th style="padding: 12px; border-bottom: 1px solid #f0f0f0;">任务 ID</th>
            <th style="padding: 12px; border-bottom: 1px solid #f0f0f0;">方法</th>
            <th style="padding: 12px; border-bottom: 1px solid #f0f0f0;">数据集</th>
            <th style="padding: 12px; border-bottom: 1px solid #f0f0f0;">M</th>
            <th style="padding: 12px; border-bottom: 1px solid #f0f0f0;">efC</th>
            <th style="padding: 12px; border-bottom: 1px solid #f0f0f0;">构建时间</th>
            <th style="padding: 12px; border-bottom: 1px solid #f0f0f0;">索引大小</th>
            <th style="padding: 12px; border-bottom: 1px solid #f0f0f0;">输出路径</th>
            <th style="padding: 12px; border-bottom: 1px solid #f0f0f0;">状态</th>
        </tr>
        <tr>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #595959;">B-001</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">HNSW-Base</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #595959;">SIFT-1M</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">16</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">200</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">18.6 min</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">5.88GB</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #8c8c8c;">outputs/hnsw/sift_M16.bin</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0;"><span style="color: #52c41a; background: #f6ffed; border: 1px solid #b7eb8f; padding: 2px 6px; border-radius: 4px;">完成</span></td>
        </tr>
        <tr>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #595959;">B-002</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">Optimized HNSW</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #595959;">Deep1B</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">32</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">200</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">102.4 min</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">34.21GB</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #8c8c8c;">outputs/hnsw/deep1b_M32.bin</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0;"><span style="color: #52c41a; background: #f6ffed; border: 1px solid #b7eb8f; padding: 2px 6px; border-radius: 4px;">完成</span></td>
        </tr>
        <tr style="background-color: #f0f5ff;">
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #1890ff; font-weight: 500;">B-003</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #1890ff; font-weight: 500;">GloVe HNSW</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #1890ff; font-weight: 500;">GloVe-840B</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #1890ff; font-weight: 500;">16</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #1890ff; font-weight: 500;">400</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #1890ff; font-weight: 500;">67.8 min</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #1890ff; font-weight: 500;">28.74GB</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #1890ff; font-weight: 500;">outputs/hnsw/glove_M16.bin</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0;"><span style="color: #1890ff; background: #e6f7ff; border: 1px solid #91d5ff; padding: 2px 6px; border-radius: 4px;">运行中</span></td>
        </tr>
        <tr>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #595959;">B-004</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">SIEVE Subindex</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #595959;">SIFT-1M</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">8</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">100</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">12.3 min</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #262626;">3.12GB</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0; color: #8c8c8c;">outputs/hnsw/sieve_M8.bin</td>
            <td style="padding: 12px; border-bottom: 1px solid #f0f0f0;"><span style="color: #52c41a; background: #f6ffed; border: 1px solid #b7eb8f; padding: 2px 6px; border-radius: 4px;">完成</span></td>
        </tr>
    </table>
    
    <div style="display: flex; justify-content: space-between; align-items: center; font-size: 13px; color: #595959; margin-top: 16px;">
        <div>共 4 条</div>
        <div style="display: flex; gap: 8px; align-items: center;">
            <select style="border: 1px solid #d9d9d9; border-radius: 4px; padding: 2px 4px; background: #fff; outline: none; color: #595959;">
                <option>10 条/页</option>
            </select>
            <div style="display: flex; gap: 4px;">
                <div style="border: 1px solid #d9d9d9; padding: 2px 8px; border-radius: 4px; cursor: pointer; color: #bfbfbf;">&lt;</div>
                <div style="border: 1px solid #1890ff; background: #1890ff; color: #fff; padding: 2px 8px; border-radius: 4px; cursor: pointer;">1</div>
                <div style="border: 1px solid #d9d9d9; padding: 2px 8px; border-radius: 4px; cursor: pointer;">&gt;</div>
            </div>
        </div>
    </div>
    """
    table_html = "".join([line.strip() for line in table_html.split("\n")])
    st.markdown(table_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

