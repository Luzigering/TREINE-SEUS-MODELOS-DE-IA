from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from google import genai
from google.genai import types
import PIL.Image
import io
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
        return "<h1>Erro: Arquivo index.html não encontrado na pasta api!</h1>"

@app.post("/api/consultar-assistente")
async def consultar_assistente(
    file: Optional[UploadFile] = File(None),
    texto: Optional[str] = Form(None)
):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API Key não configurada no Vercel")

    
    if not file and not texto:
        raise HTTPException(status_code=400, detail="Envie uma imagem, um comando de texto, ou ambos.")

    try:
        client = genai.Client(api_key=API_KEY)
        
        # Área para trabalhar a personalização POSTERIORMENTE
        instrucao_sistema = """Você é um Especialista em Operações sênior e um assistente de IA altamente capacitado.
        Sua especialidade envolve:
        - Análise de eficiência operacional, logística e supply chain.
        - Auditoria e leitura técnica de notas fiscais, cupons, contratos e recibos.
        - Otimização de processos, controle de custos e gestão de tempo.
        - Resolução de gargalos operacionais.
        
        Regras de Resposta:
        1. Seja direto, analítico e extremamente profissional.
        2. Use formatação Markdown (negritos, listas, tópicos) para organizar sua resposta e facilitar a leitura, mas mantendo tom humano e se limitando a 4 frases sempre que possível; (está explicando, mas deve ser direto e didático;)
        3. Se o usuário enviar uma nota fiscal (imagem), extraia os dados principais (Estabelecimento, Data, Total, Itens, NUMERO DO PEDIDO, NOME DO CLIENTE, CANAL (MUITO IMPORTANTE, SE CONSTAR, IFOOD, RAPPI, 99, SE Não constar entenda que é um pedido efetivado no app da propria empresa, mas que a logistica para o pedido sempre será do ifood, então deve se basear nas politicas dessa plataforma no que se referir á logistica, politicas afins)) de forma limpa e faça uma breve análise operacional se houver contexto.
        ou seja, a informação crucial para o desenvolvimento da resposta é o canal em que o pedido foi efetivado constatado na nota, então se o prompt não contemplar esse detalhe, deve sempre questionar o canal de efetivação do pedido para assim personalizar a resposta de acordo com o canal e suas politicas á respeito de pedidos;
        4. Se o usuário fizer uma pergunta, aja como um consultor sênior entregando um plano de ação ou resposta técnica.
        5. Se o usuário solicitar: Numero de telefone de cliente e não for identificado que o pedido foi via canal "Ifood" (que não disponibilizam por conta da LGPD), seja breve e apenas retorne "Olá, um instante, irei verificar o número no sistema";
        6. Se o usuário solicitar: reenvio de item informe também nesse caso: "Olá, um instante, irei verificar o reenvio aqui pelo sistema;";
        """

        conteudos_para_gemini = []
        
        if texto:
            conteudos_para_gemini.append(texto)
            
        if file:
            contents = await file.read()
            image = PIL.Image.open(io.BytesIO(contents))
            conteudos_para_gemini.append(image)

    
        response = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=conteudos_para_gemini,
            config=types.GenerateContentConfig(
                system_instruction=instrucao_sistema,
                temperature=0.3 
            )
        )
  
        return {"sucesso": True, "resposta": response.text}
        
    except Exception as e:
        return {"sucesso": False, "erro": str(e)}