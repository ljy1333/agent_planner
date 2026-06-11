import streamlit as st
from agent import chat_with_rag,reset_history,nodes_to_timeline_image,generate_calendar_file
from prompts import FIRST_ROUND_PROMPT
from datetime import datetime  
st.set_page_config(page_title="个人规划参谋", page_icon="🎯")
st.title("    🎯 个人发展规划助手")
# ==================== 根据状态显示侧边栏内容 ====================
# 初始化
if "show_settings" not in st.session_state:
    st.session_state.show_settings = False
if "show_Instructions" not in st.session_state:
    st.session_state.show_Instructions = False
if "session_id" not in st.session_state:
    st.session_state.session_id = "default_user"
if "display_messages" not in st.session_state:
    st.session_state.display_messages = []
# ==================== 顶部控制栏 ====================
if st.button("🔄 重新开始"):
    reset_history(st.session_state.session_id)
    st.session_state.display_messages = []
    st.session_state.show_sidebar = False
    st.rerun()
with st.sidebar:
    st.header("关于")
    # 按钮行：并排放两个按钮
    col1, col2= st.columns(2)
    with col1:
        if st.button("📖 说明", use_container_width = True):
            st.session_state.show_Instructions = True
            st.session_state.show_settings = False 
    with col2:
        if st.button("⚙️ 设置", use_container_width = True):
            st.session_state.show_settings = True
            st.session_state.show_Instructions = False
    # 根据按钮选择显示不同内容
    if st.session_state.show_settings:
        st.markdown("### 设置")
        # 会话ID管理
        current_session_id = st.session_state.session_id
        new_session_id = st.text_input(
            "会话ID",
            value = current_session_id,
            help="修改会话ID可以切换到不同的对话历史"
        )
        # 如果会话ID改变，更新状态
        if new_session_id != current_session_id:
            st.session_state.session_id = new_session_id
            st.session_state.display_messages = []  # 清空显示的消息
            st.rerun()
        st.divider()
        st.markdown("### 📁 上传参考文件")
        st.caption("上传 PDF 或文本文件，AI 将基于文件内容制定计划")
        uploaded_file = st.file_uploader(
            "选择文件",
            type=["pdf", "txt", "md"],
            label_visibility="collapsed"
        )
        if uploaded_file is not None:
            with st.spinner("处理文件中..."):
                try:
                    from agent import process_uploaded_file
                    msg = process_uploaded_file(
                        uploaded_file.read(),
                        uploaded_file.name,
                        st.session_state.session_id
                    )
                    st.success(msg)
                except Exception as e:
                    st.error(f"处理失败: {e}")
        if st.button("📅 生成时间轴图片"):
            with st.spinner("正在生成时间轴..."):
                path = nodes_to_timeline_image(st.session_state.session_id)  # 传递 session_id
                if path and "无法" not in path:
                    st.image(path, caption="你的规划时间轴")
                else:
                    st.error(path)
        # 显示当前会话信息
        st.divider()
        if 'chat_histories' in st.session_state:
            if st.session_state.session_id in st.session_state.chat_histories:
                history = st.session_state.chat_histories[st.session_state.session_id]
                st.caption(f"📝 当前会话消息数: {len(history.messages)}")
        st.divider()
        st.markdown('### 导出到日历 ')
        if  st.button("生成日历文件",use_container_width = True):
            with st.spinner("正在生成日历..."):
                try:
                    cal_path = generate_calendar_file(st.session_state.session_id)
                    with open(cal_path,"rb") as f:
                        st.download_button(
                            label = "下载日历文件(.ics)",
                            data = f,
                            file_name = cal_path.split("/")[-1],
                            mime = "text/calendar"
                        )
                    st.success("日历文件已生成！请下载后导入手机或电脑日历")
                except Exception as e:
                    st.error(f"生成失败：{e}")
                    st.info("请先在聊天中完成规划，在导出日历")
    # 显示说明内容
    elif st.session_state.show_Instructions:
        st.markdown(FIRST_ROUND_PROMPT)
        
        # 始终显示当前会话ID（不在设置里也显示）
        st.divider()
        st.caption(f"当前会话: `{st.session_state.session_id}`")

# ==================== 显示历史消息 ====================
for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
# ==================== 用户输入 ====================
user_input = st.chat_input("说说你的目标...")
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.display_messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            reply = chat_with_rag(user_input, st.session_state.session_id)
        st.markdown(reply)
    st.session_state.display_messages.append({"role": "assistant", "content": reply})