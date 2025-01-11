import os
import tempfile
import pandas as pd
import pytesseract
from PIL import Image
import docx
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from pypdf.errors import PdfReadError
from openai import AuthenticationError, BadRequestError




# Alterar o estilo da página com CSS
st.markdown(
    """
    <style>
    body {
        background-color: #1A1A1D;
        color: #A64D79;
    }
    .stButton > button {
        background-color: #FF204E;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Adicionar o título personalizado com HTML


st.markdown(
    """
    <h1 style='text-align: center; color:white;'>
        By: <span style='color: white;'>Honacleon Junior</span>
    </h1>
    """,
    unsafe_allow_html=True
)

# Adicionar a imagem no cabeçalho
# image_url = "https://cienciadosdados.com/images/CINCIA_DOS_DADOS_4.png"
# st.image(image_url, use_container_width=True)

# Adicionar o nome do aplicativo
st.subheader("Perguntas e Respostas com IA - PLN usando LangChain")

# Componentes interativos
file_input = st.file_uploader("Faça upload de um arquivo", type=['pdf', 'txt', 'csv', 'docx', 'jpeg', 'png'])
openaikey = st.text_input("Digite sua chave de API da OpenAI", type='password')
prompt = st.text_area("Digite suas perguntas", height=160)
run_button = st.button("Executar!")

select_k = st.slider("Número de trechos relevantes", min_value=1, max_value=5, value=2)
select_chain_type = st.radio("Tipo de cadeia", ['stuff', 'map_reduce', "refine", "map_rerank"])

# Função para carregar documentos
def load_document(file_path, file_type):
    if file_type == 'application/pdf':
        loader = PyPDFLoader(file_path)
        return loader.load()
    elif file_type == 'text/plain':
        loader = TextLoader(file_path)
        return loader.load()
    elif file_type == 'text/csv':
        df = pd.read_csv(file_path)
        return [{"page_content": df.to_string()}]
    elif file_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        doc = docx.Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return [{"page_content": "\n".join(full_text)}]
    elif file_type in ['image/jpeg', 'image/png']:
        text = pytesseract.image_to_string(Image.open(file_path))
        return [{"page_content": text}]
    else:
        st.error("Tipo de arquivo não suportado.")
        return None

# Função de perguntas e respostas
def qa(file_path, file_type, query, chain_type, k):
    try:
        documents = load_document(file_path, file_type)
        if not documents:
            return None
        
        # Dividir os documentos em trechos
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        
        # Selecionar quais embeddings usar
        embeddings = OpenAIEmbeddings()
        
        # Criar o banco de vetores para usar como índice
        db = Chroma.from_documents(texts, embeddings)
        
        # Expor esse índice em uma interface de recuperação
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
        
        # Criar uma cadeia para responder às perguntas
        qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model="gpt-4o-mini"), 
            chain_type=chain_type, 
            retriever=retriever, 
            return_source_documents=True
        )
        result = qa.run(query)
        return result
    except PdfReadError as e:
        st.error(f"Erro ao ler o arquivo PDF: {e}")
        return None
    except AuthenticationError as e:
        st.error(f"Erro de autenticação: {e}")
        return None
    except BadRequestError as e:
        st.error(f"Erro de requisição inválida: {e}")
        return None

# Função para exibir o resultado no Streamlit
def display_result(result):
    if result:
        st.markdown("### Resultado:")
        st.write(result)

# Execução do app
if run_button and file_input and openaikey and prompt:
    with st.spinner("Executando QA..."):
        # Salvar o arquivo em um local temporário
        temp_file_path = os.path.join(tempfile.gettempdir(), file_input.name)
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(file_input.read())

        # Configurar a chave de API do OpenAI
        os.environ["OPENAI_API_KEY"] = openaikey

        # Verificar se a chave de API é válida
        try:
            # Testar a chave de API com uma chamada simples
            embeddings = OpenAIEmbeddings()
            embeddings.embed_documents(["test"])
        except AuthenticationError as e:
            st.error(f"Chave de API da OpenAI inválida: {e}")
        else:
            # Executar a função de perguntas e respostas
            result = qa(temp_file_path, file_input.type, prompt, select_chain_type, select_k)
            # Exibir o resultado
            display_result(result)