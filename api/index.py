from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
from google import genai
from google.genai import types
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel 
import mimetypes
from vercel.blob import AsyncBlobClient
import httpx
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

API_KEY = os.environ.get("GEMINI_API_KEY")
MONGO_URI = os.environ.get("MONGO_URI")
client = AsyncIOMotorClient(MONGO_URI) if MONGO_URI else None
db = client.treinamento_ia if client is not None else None
personas_collection = db.personas if db is not None else None
class Persona(BaseModel):
    nome: str
    prompt: str
    file_url: Optional[str] = None
@app.get("/", response_class=HTMLResponse)
def home():
    try:
        with open(os.path.join(os.path.dirname(__file__), "index.html"), "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Erro: Ficheiro index.html não encontrado na pasta api!</h1>"
@app.get("/api/personas")
async def get_personas():
    if personas_collection is None:
        return {"sucesso": False, "erro": "Banco de dados não configurado"}
    
    cursor = personas_collection.find({}, {"_id": 0})
    personas = await cursor.to_list(length=100)
    return {"sucesso": True, "personas": personas}

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        client = AsyncBlobClient()
        content = await file.read()
        
       
        blob = await client.put(
            file.filename, 
            content, 
            access="private",
            add_random_suffix=True 
        )
        
        return {"sucesso": True, "url": blob.url}
    except Exception as e:
        return {"sucesso": False, "erro": str(e)}
    
@app.post("/api/personas")
async def create_persona(persona: Persona):
    if personas_collection is None:
        return {"sucesso": False, "erro": "Banco de dados não configurado"}
    
    await personas_collection.insert_one(persona.model_dump())
    return {"sucesso": True, "mensagem": "Persona salva na nuvem!"}
import mimetypes 

@app.post("/api/gemma-chat")
async def gemma_chat(
    files: Optional[List[UploadFile]] = File(None),
    system_prompt: Optional[str] = Form(""),
    user_query: Optional[str] = Form(""),
    persona_file_url: Optional[str] = Form(None),
    persona_nome: Optional[str] = Form(None)
):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API Key não configurada no Vercel/Ambiente")
        
    if persona_nome and personas_collection is not None:
        persona_db = await personas_collection.find_one({"nome": persona_nome})
        if persona_db:
            system_prompt = persona_db.get("prompt", system_prompt)
            persona_file_url = persona_db.get("file_url", persona_file_url)
        else:
            return {"sucesso": False, "erro": f"O modelo '{persona_nome}' não foi encontrado no banco de dados."}
            
    if not files and not user_query and not persona_file_url:
        raise HTTPException(status_code=400, detail="Envia documentos, seleciona uma persona com treino ou faz uma pergunta.")

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
                
        
    
        if persona_file_url:
            try:
                blob_token = os.environ.get("BLOB_READ_WRITE_TOKEN")
                
               
                headers = {"Authorization": f"Bearer {blob_token}"}
                
                
                async with httpx.AsyncClient(follow_redirects=True) as http_client:
                    resp = await http_client.get(persona_file_url, headers=headers)
                    
                    if resp.status_code == 200:
                        file_bytes = resp.content
                        
                       
                        url_limpa = persona_file_url.split("?")[0]
                        mime_type, _ = mimetypes.guess_type(url_limpa)
                        if not mime_type:
                            mime_type = "application/pdf" 
                            
                        blob_part = types.Part.from_bytes(
                            data=file_bytes,
                            mime_type=mime_type,
                        )
                        conteudos.append(blob_part)
                        
                        
                        conteudos.append(
                            "O documento fornecido contém as suas DIRETRIZES DE PERSONA, REGRAS E BASE DE CONHECIMENTO EXCLUSIVA. "
                            "Você DEVE seguir rigidamente o tom de voz e as regras definidas nele. "
                            "NÃO use nenhum conhecimento prévio que conflite com este documento."
                        )
                    else:
                        print(f"Erro Vercel: {resp.status_code} - {resp.text}")
                        
            except Exception as e:
                print(f"Aviso: Erro ao tentar descarregar o ficheiro da Persona na nuvem. Erro: {e}")
       
        if user_query:
            conteudos.append(f"Pergunta do utilizador: {user_query}")
        else:
            conteudos.append("Por favor, faz um resumo dos documentos fornecidos com base na tua especialidade.")

        if not system_prompt:
            system_prompt = "Você é um assistente de IA focado na extracção e análise de documentos fornecidos pelo utilizador. Responde com clareza, seguindo á risca suas diretrizes;."

        response = client.models.generate_content(
            model="gemma-2-9b-it", 
            contents=conteudos,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.3 
            )
        )
        return {"sucesso": True, "resposta": response.text}
        
    except Exception as e:
        return {"sucesso": False, "erro": str(e)}



@app.delete("/api/personas")
async def delete_persona(nome: str):
    if personas_collection is None:
        return {"sucesso": False, "erro": "Banco de dados não configurado"}
    
    persona = await personas_collection.find_one({"nome": nome})
    if not persona:
        return {"sucesso": False, "erro": "Persona não encontrada"}
        
    if persona.get("file_url"):
        try:
            from vercel.blob import AsyncBlobClient 
            blob_client = AsyncBlobClient()
            await blob_client.delete(persona["file_url"])
        except Exception as e:
            print(f"Aviso: Não foi possível apagar o PDF na Vercel. Erro: {e}")
            
    await personas_collection.delete_one({"nome": nome})
    return {"sucesso": True, "mensagem": "Persona apagada com sucesso!"}
