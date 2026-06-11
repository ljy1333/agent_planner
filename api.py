from fastapi import FastAPI
from pydantic import BaseModel
from agent import chat,reset_history,generate_timeline_nodes,nodes_to_timeline_image
app = FastAPI(title="个人规划参谋 API")

class ChatRequest(BaseModel):
    user_input : str
    session_id : str = "default_user"

class ResetRequest(BaseModel):
    session_id : str = "default_user"

@app.get("/")
def root():
    return{"message": "个人规划参谋 API 运行中"}

@app.post("/chat")
def api_chat(req: ChatRequest):
    reply = chat (req.user_input, req.session_id)
    return {"reply" : reply,"session_id": req.session_id}
@app.post("/reset")
def api_reset(reg:ResetRequest):
    reset_history(reg.session_id)
    return{"status":"ok","session_id":reg.session_id}