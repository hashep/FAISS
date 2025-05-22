import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory

openai_api_key = st.secrets["openai"]["api_key"]

@st.cache_resource
def load_and_split_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load_and_split()

@st.cache_resource
def create_vector_store(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_docs = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(split_docs, OpenAIEmbeddings(model='text-embedding-3-small', openai_api_key=openai_api_key))
    vectorstore.save_local("faiss_index")
    return vectorstore

@st.cache_resource
def get_vectorstore(docs):
    if os.path.exists("faiss_index"):
        return FAISS.load_local("faiss_index", OpenAIEmbeddings(model='text-embedding-3-small', openai_api_key=openai_api_key))
    else:
        return create_vector_store(docs)

@st.cache_resource
def initialize_components(selected_model, file_path):
    pages = load_and_split_pdf(file_path)
    vectorstore = get_vectorstore(pages)
    retriever = vectorstore.as_retriever()

    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ])

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Keep the answer perfect. please use imogi with the answer. 대답은 한국어로 하고, 존댓말을 써줘.\n\n{context}"),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ])

    llm = ChatOpenAI(model=selected_model, openai_api_key=openai_api_key)
    retriever_with_history = create_history_aware_retriever(llm, retriever, contextualize_prompt)
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(retriever_with_history, qa_chain)
    return rag_chain

st.header("PDF 기반 Q&A 챗봇 💬 📚")
uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type="pdf")

if uploaded_file:
    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_file.read())

    model_option = st.selectbox("GPT 모델 선택", ("gpt-4o-mini", "gpt-3.5-turbo-0125"))
    rag_chain = initialize_components(model_option, "uploaded.pdf")
    chat_history = StreamlitChatMessageHistory(key="chat_messages")

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: chat_history,
        input_messages_key="input",
        history_messages_key="history",
        output_messages_key="answer",
    )

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "PDF 문서에 대해 무엇이든 물어보세요!"}]

    for msg in chat_history.messages:
        st.chat_message(msg.type).write(msg.content)

    if prompt := st.chat_input("질문을 입력하세요"):
        st.chat_message("human").write(prompt)
        with st.chat_message("ai"):
            with st.spinner("Thinking..."):
                config = {"configurable": {"session_id": "user-session"}}
                response = conversational_rag_chain.invoke({"input": prompt}, config)
                answer = response['answer']
                st.write(answer)
                with st.expander("참고 문서 보기"):
                    for doc in response['context']:
                        st.markdown(doc.metadata.get('source', ''), help=doc.page_content)
