import os
import streamlit as st
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "deepseek-ai/DeepSeek-V3"
HF_BASE_URL = "https://router.huggingface.co/v1"


@st.cache_resource
def init_chain():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=os.getenv("PINECONE_INDEX"),
        embedding=embeddings,
    )

    prompt_template = """
Use the following context to answer the user's question about stocks, companies, markets, or financial data.

Rules:
- Only use the information provided in the context.
- If the answer is not in the context, say "I don't have enough information to answer that."
- Provide clear and concise explanations.
- Do not give financial advice or investment recommendations.

Context: {context}
User Question: {question}

Answer:
"""
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
    )

    llm = ChatOpenAI(
        model=LLM_MODEL,
        api_key=os.getenv("HF_TOKEN"),
        base_url=HF_BASE_URL,
        temperature=0.7,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
    )


# --- Streamlit UI ---
st.set_page_config(page_title="Finance ChatBot", page_icon="📈")
st.title("📈 Finance ChatBot")

qa_chain = init_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask a question about stocks or markets..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = qa_chain.invoke({"query": question})
            answer = result["result"]
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})