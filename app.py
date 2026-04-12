# ─── IMPORTS ───
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage,AIMessage 
import tempfile
import os
from dotenv import load_dotenv

# ─── LOAD API KEY ───
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ─── PAGE SETUP ───
st.set_page_config(
    page_title="RAG Document Chatbot",
    page_icon="📄",
    layout="wide"
)

st.title("📄 RAG Document Chatbot")
st.markdown("**Upload a PDF and ask questions about it using Google Gemini AI**")
st.divider()

# ─── CACHE PROCESSING (IMPORTANT FOR PERFORMANCE) ───
@st.cache_resource
def process_pdf(file_path):

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GEMINI_API_KEY
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vectorstore, len(chunks)

# ─── FILE UPLOAD ───
uploaded_file = st.file_uploader("Upload your PDF document", type="pdf")

if uploaded_file is not None:

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    try:
        with st.spinner("Processing document..."):
            vectorstore, chunk_count = process_pdf(tmp_path)

        st.success(f"✅ Document processed! {chunk_count} chunks created.")

        # ─── CREATE RETRIEVER ───
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3}
        )

        # ─── CREATE LLM ───
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=GEMINI_API_KEY,
            temperature=0.3
        )

        # ─── PROMPT ───
        prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant.    

        Use the chat history and context below to answer the question.

        Chat History:
        {chat_history}

        Context:
        {context}

        Question:
        {input}
        """)

        # ─── LCEL CHAIN (NEW LANGCHAIN v1 STYLE) ───
        # qa_chain = (
        #     {
        #         "context": retriever,
        #         "input": RunnablePassthrough(),
        #         "chat_history": lambda x: st.session_state.chat_history
        #     }
        #     | prompt
        #     | llm
        #     | StrOutputParser()
        # )

        # ─── CHAT HISTORY ─── 
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Memory for LLM 
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []    

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # ─── USER INPUT ───
        question = st.chat_input("Ask a question about your document...")

        if question:
            # Store user message
            st.session_state.messages.append({
                "role": "user",
                "content": question
            })

            with st.chat_message("user"):
                st.write(question)

            # Get answer
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    docs = retriever.invoke(question)
                    context = "\n\n".join([doc.page_content for doc in docs]) 

                    if not context.strip():
                        answer = "I could not find relevant information in the document."
                    else: 
                        formatted_prompt = prompt.format(
                            context=context,
                            input=question,
                            chat_history=st.session_state.chat_history)
                        
                        response = llm.invoke(formatted_prompt)
                        answer = response.content

                    st.write(answer) 
                    with st.expander("📄 Sources"):
                        for i, doc in enumerate(docs):
                            page = doc.metadata.get("page", "Unknown")

                            if page is not None: 
                                st.markdown(f"**Page {page + 1}**")
                            else:
                                st.markdown("**Page Unknown**")    
                            st.write(doc.page_content[:300] + "...")

                    # Save conversation for memory
                    st.session_state.chat_history.append(HumanMessage(content=question))
                    st.session_state.chat_history.append(AIMessage(content=answer))

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

    finally:
        # Clean temp file safely
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

else:
    st.info("👆 Please upload a PDF document to get started!")