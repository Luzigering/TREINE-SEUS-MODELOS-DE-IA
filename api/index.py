from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
from google import genai
from google.genai import types
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"], 
    allow_headers=["*"],
)

API_KEY = os.environ.get("GEMINI_API_KEY")

@app.get("/", response_class=HTMLResponse)
def home():
    try:
        with open(os.path.join(os.path.dirname(__file__), "index.html"), "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Erro: Ficheiro index.html não encontrado na pasta api!</h1>"

@app.post("/api/gemma-chat")
async def gemma_chat(
    files: Optional[List[UploadFile]] = File(None),
    system_prompt: Optional[str] = Form(""),
    user_query: Optional[str] = Form("")
):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API Key não configurada no Vercel/Ambiente")

    if not files and not user_query:
        raise HTTPException(status_code=400, detail="Envia documentos ou uma pergunta.")

    try:
        client = genai.Client(api_key=API_KEY)
        
        conteudos = []
        
        if files:
            for file in files:
                contents = await file.read()
                file_part = types.Part.from_bytes(
                    data=contents,
                    mime_type=file.content_type,
                )
                conteudos.append(file_part)
                
        if user_query:
            conteudos.append(f"Pergunta do utilizador com base nos documentos acima: {user_query}")
        else:
            conteudos.append("Por favor, faz um resumo dos documentos fornecidos.")

        if not system_prompt:
            system_prompt = "És um assistente de IA focado na extracção e análise de documentos fornecidos pelo utilizador. Responde com clareza usando formatação Markdown."

        response = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=conteudos,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.3 
            )
        )
  
        return {"sucesso": True, "resposta": response.text}
        
    except Exception as e:
        return {"sucesso": False, "erro": str(e)}