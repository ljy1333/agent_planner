from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import config
from prompts import SYSTEM_PROMPT, COMPRESS_SYSTEM_PROMPT,TIMELINE_NODES_PROMPT,TIMELINE_TO_IMAGE_PROMPT
import base64
import requests
from datetime import datetime
import streamlit as st
from ics import Calendar,Event
import os
import tempfile
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
model = ChatOpenAI(
    model = config.MODEL_NAME,
    base_url=config.BASE_URL,
    api_key=config.API_KEY,
    temperature=config.TEMPERATURE
)
_embeddings = None
def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(
            model="BAAI/bge-m3",
            base_url=config.BASE_URL,
            api_key=config.API_KEY
        )
    return _embeddings
def process_uploaded_file(file_bytes: bytes, file_name: str, session_id: str = "default_user") -> str:
    # 1. 保存临时文件
    suffix = os.path.splitext(file_name)[1].lower()#分割文件名获取列表以获得后缀
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    
    # 2. 读取文件内容
    if suffix == ".pdf":
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
    elif suffix in [".txt", ".md"]:
        with open(tmp_path, "r", encoding="utf-8") as f:
            text = f.read()
        docs = [Document(page_content=text)]
    else:
        os.unlink(tmp_path)
        raise Exception(f"不支持的文件类型: {suffix}")
    
    # 3. 删除临时文件
    os.unlink(tmp_path)
    
    # 4. 切成小段
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    #返回列表，每个元素都是一个Document对象
    chunks = splitter.split_documents(docs)
    print(chunks)
    
    # 5. 存入向量数据库
    embeddings = get_embeddings()
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db",
        collection_name=f"user_{session_id}"
    )
    
    return f"已处理 `{file_name}`，共 {len(chunks)} 个文本片段"
def search_docs(query: str , session_id , k = 3):
    try:
        embeddings = get_embeddings()
        vector_store = Chroma(
            persist_directory = "./chroma_db",
            embedding_function = embeddings,
            collection_name= f"user_{session_id}"
        )
        results = vector_store.similarity_search(query,k = k)
        if not results:
            return ""
        content = ""
        for i,doc in enumerate(results):# i 是从0开始的标号
            content += f"参考片段{i+1}\n{doc.page_content}\n\n"
        return content
    except Exception:
        return ""
def chat_with_rag(user_input:str,session_id:str = "default_user") -> str:
    docs = search_docs(user_input,session_id)
    if docs:
        enhanced_input = f"""【用户上传的参考资料】
{docs}
【用户当前问题】
{user_input}
请基于以上参考资料回答问题。如果参考资料与问题无关，忽略参考资料直接回答。"""
    else:
        # 没搜到：原样用用户的问题
        enhanced_input = user_input
    return chat(enhanced_input,session_id)
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if 'chat_histories' not in st.session_state:
        st.session_state.chat_histories = {}
    if session_id not in st.session_state.chat_histories:
        st.session_state.chat_histories[session_id] = InMemoryChatMessageHistory()
    return st.session_state.chat_histories[session_id]
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
def get_system_prompt_with_time(prompt) -> str:
    now = datetime.now()
    time_str = now.strftime("%Y年%m月%d日 %H:%M")
    weekday = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][now.weekday()]
    return prompt +f"\n\n## 当前时间\n现在是{time_str}，{weekday}。请基于此时间为用户制定计划。"
prompt = ChatPromptTemplate.from_messages([
    ("system", get_system_prompt_with_time(SYSTEM_PROMPT)),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])
##接受一个字典返回一个模范的列表
chain = prompt | model
with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,#工厂函数，返回信息的
    input_messages_key="input",#指定输入字典中用户输入内容的键名
    history_messages_key="history",#通过get_session_history获取的InMemoryChatMessageHistory.message
                                   #提取所有信息以转换为新字典“history”的value，传递给链路中的prompt
)
# RunnableWithMessageHistory 包装原有 chain，自动从 get_session_history 获取历史消息，
# 将历史消息以 "history" 键注入到输入字典中，再调用原 chain。
# 最终通过 invoke 返回模型的响应内容。
def chat(user_input: str, session_id: str = "default_user") -> str:
    history = get_session_history(session_id)
    if len(history.messages) > config.MAX_HISTORY_LENGTH:
        compress_history(session_id)
    response = with_message_history.invoke(
        {"input": user_input},#对应RUNABLE里的input字典的值
        config = {"configurable": {"session_id": session_id}}#搜寻id
    )
    return response.content
def compress_history(session_id: str = "default_user"):
    if session_id not in st.session_state.chat_histories:
        return
    
    history = st.session_state.chat_histories[session_id]
    messages = history.messages  # ← 取 .messages 列表
    
    if len(messages) <= config.MAX_HISTORY_LENGTH:
        return
    
    # 保留最近消息，压缩前面的
    recent = messages[-config.KEEP_RECENT:]
    to_compress = messages[:-config.KEEP_RECENT]
    
    # 拼接为纯文本
    compress_text = ""
    for m in to_compress:
        if m.type == "human":
            role = "用户"
        elif m.type == "ai":
            role = "助手"
        else:
            role = "系统"  # 或跳过
        compress_text += f"{role}: {m.content}\n"
    
    # 用原生 llm.invoke，不走 with_message_history（避免循环写入）
    summary_response = model.invoke([
        SystemMessage(content=COMPRESS_SYSTEM_PROMPT),
        HumanMessage(content=compress_text)
    ])

    summary = summary_response.content
    
    # 重建消息列表
    history.clear()
    history.add_message(AIMessage(content=f"[历史对话摘要]: {summary}"))
    for m in recent:
        history.add_message(m)
def reset_history(session_id: str = "default_user"):
    """重置指定用户的对话历史"""
    if session_id in st.session_state.chat_histories:
        st.session_state.chat_histories[session_id] = InMemoryChatMessageHistory()
def generate_image(prompt: str, save_path: str = "generated_image.png") -> str:
    url = "https://api.siliconflow.cn/v1/images/generations"
    headers = {
        "Authorization": f"Bearer {config.API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "Qwen/Qwen-Image",
        "prompt":prompt,
        "n": 1,
        "size": "1024x1024"
    }
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        data = response.json()
        image_url = data["data"][0]["url"]
        img_data = requests.get(image_url).content
        with open(save_path, "wb") as f:
            f.write(img_data)
        return save_path
    else:
        raise Exception(f"图片生成失败: {response.text}")
def generate_timeline_nodes(session_id: str = "default_user") -> str:
    """从对话历史提炼时间轴节点"""
    if session_id not in st.session_state.chat_histories:
        return "无对话历史"
    
    history = st.session_state.chat_histories[session_id]
    messages = history.messages
    
    full_text = ""
    for m in messages:
        role = "用户" if m.type == "human" else "助手"
        full_text += f"{role}: {m.content}\n"
    
    response = model.invoke([
        SystemMessage(content=TIMELINE_NODES_PROMPT),
        HumanMessage(content=full_text)
    ])
    return response.content
def nodes_to_timeline_image(session_id: str = "default_user") -> str:
    """把时间轴节点转成图片描述，然后生成图片"""
    # 第一步：生成图片描述
    node = generate_timeline_nodes(session_id)
    if node == "无对话历史":
        return "无对话历史，无法生成时间轴"
    prompt_response = model.invoke([
        SystemMessage(content=TIMELINE_TO_IMAGE_PROMPT),
        HumanMessage(content=generate_timeline_nodes(session_id))
    ])
    image_prompt = prompt_response.content
    
    # 第二步：调用图片生成
    image_path = generate_image(image_prompt, "timeline.png")
    return image_path
def generate_calendar_file(session_id: str = "default_user") -> str:
    nodes_text = generate_timeline_nodes(session_id)
    if nodes_text == "无对话历史":
        raise Exception("还没有生成规划，请先完成规划再导出日历")
    parse_prompt = f"""把以下规划内容解析为JSON格式，用于生成日历。

        {nodes_text}

        输出格式（严格JSON）：
        {{
            "goal": "总目标",
            "phases": [
                {{
                    "name": "阶段名称",
                    "start": "YYYY-MM-DD",
                    "end": "YYYY-MM-DD",
                    "tasks": ["任务1", "任务2"]
                }}
            ]
        }}
        当前日期是{datetime.now().strftime("%Y-%m-%d")}。如果表格里写的是相对时间（如"第1-4周"），请以当前日期为起点换算为具体日期。
只输出JSON，不要其他文字。"""
    
    response = model.invoke([HumanMessage(content=parse_prompt)])
    import json
    data = json.loads(response.content)
    
    # 3. 创建日历
    cal = Calendar()
    now = datetime.now()
    
    for phase in data["phases"]:
        start_date = datetime.strptime(phase["start"], "%Y-%m-%d")
        end_date = datetime.strptime(phase["end"], "%Y-%m-%d")
        
        # 阶段起止事件
        event = Event()
        event.name = f"【{data['goal']}】{phase['name']} 开始"
        event.begin = start_date
        event.make_all_day()
        event.description = "\n".join([f"• {t}" for t in phase.get("tasks", [])])
        cal.events.add(event)
        # 里程碑提醒（阶段中点）
        mid_date = start_date + (end_date - start_date) / 2
        event = Event()
        event.name = f"📌 {phase['name']} 中期检查"
        event.begin = mid_date
        event.make_all_day()
        event.description = "检查进度，对照里程碑评估完成情况"
        cal.events.add(event)
    
    # 4. 保存文件
    file_path = f"plan_{now.strftime('%Y%m%d_%H%M%S')}.ics"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(cal.serialize())
    
    return file_path