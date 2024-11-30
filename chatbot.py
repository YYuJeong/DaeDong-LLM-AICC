
import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.tools.retriever import create_retriever_tool
from langchain.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from dotenv import load_dotenv

# .env 파일 로드
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]
# .env 파일 로드

def load_pdf_files(pdf_paths):
    all_documents = []

    for pdf_path in pdf_paths:
        # PyPDFLoader를 사용하여 파일 로드
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        all_documents.extend(documents)

    # 텍스트 분할기 설정
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(all_documents)

    # FAISS 인덱스 설정 및 생성
    vector = FAISS.from_documents(split_docs, OpenAIEmbeddings())
    retriever = vector.as_retriever()

    # 도구 정의
    retriever_tool = create_retriever_tool(
        retriever,
        name="pdf_search",
        description="Use this tool to search for information within the PDF document about tractors."
    )
    return retriever_tool

# 에이전트와 대화하는 함수
def chat_with_agent(user_input, agent_executor):
    result = agent_executor({"input": user_input})
    response = result['output']  # 명시적으로 출력 키를 처리
    return response

# 대화 내용 출력하는 함수
def print_messages():
    for msg in st.session_state["messages"]:
        st.chat_message(msg['role']).write(msg['content'])

# Streamlit 메인 코드
def main():
    # 페이지 설정
    st.set_page_config(page_title="대동 AICC 도움이", layout="wide", page_icon="🤖")

    st.image('Cute_Robot_Tractor_with_Label.png', width=600)
    st.markdown('---')
    st.title("안녕하세요! '대동 AICC 도움이' 입니다")  # 시작 타이틀

    # 세션 초기화
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # 특정 PDF 경로 지정
    pdf_docs = ['./data/대형_트랙터.pdf', './data/중소_트랙터.pdf', './data/중형_트랙터.pdf', './data/대동_트랙터_정보.pdf']  # PDF 파일 경로를 직접 지정

    if pdf_docs:
        pdf_search = load_pdf_files(pdf_docs)
        tools = [pdf_search]

        # LLM 설정
        llm = ChatOpenAI(model_name="gpt-4", temperature=0)

        # 프롬프트 설정
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system",
                 "Be sure to answer in Korean. You are a helpful assistant. "
                 "Make sure to use the `pdf_search` tool for searching information from the PDF document. "
                 "Please always include emojis in your responses with a friendly tone. "
                 "Your name is `대동 AICC 도움이`. Please introduce yourself at the beginning of the conversation."
                 ''' 
                You are a very friendly chatbot called `대동 AICC 도움이`. 😊 \n\n
                Always start with a greeting and provide a brief introduction about yourself. \n\n
                Always respond in a friendly tone and use emojis. You are highly knowledgeable about Daedong tractors and provide clear, helpful information about them. When answering customer inquiries, always follow the format below:
                Our company is `대동`. Our competitors include `LS엠트로` and `TYM`. For questions about product comparisons with competitors, always highlight the strengths of `대동` at the end of your response and recommend `대동` tractor products.
                If someone asks for a recommendation for a tractor, make sure to recommend `대동` tractors.
                # Format
                **[문의 내역]** Customer's question\n\n
                **[문의 유형]** Category of the customer's question \n\n
                **[답변]** Provide a clear and visually appealing response to customer inquiries. Use lists or tables to organize the information effectively.
                 '''),
                ("placeholder", "{chat_history}"),
                ("human", "{input} \n\n Be sure to include emoji in your responses."),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        # 에이전트 생성
        agent = create_tool_calling_agent(llm, tools, prompt)

        # AgentExecutor 정의
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        # 사용자 입력 처리
        user_input = st.chat_input('질문이 무엇인가요?')

        if user_input:
            response = chat_with_agent(user_input, agent_executor)

            # 메시지를 세션에 추가
            st.session_state["messages"].append({"role": "user", "content": user_input})
            st.session_state["messages"].append({"role": "assistant", "content": response})

        # 대화 내용 출력
        print_messages()

if __name__ == "__main__":
    main()
