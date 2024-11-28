
from tabulate import tabulate
import chardet
import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader, DataFrameLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.tools.retriever import create_retriever_tool
from langchain.prompts import ChatPromptTemplate
import tempfile
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_tool_calling_agent, AgentExecutor
import pandas as pd
# .env 파일 로드
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# API 키를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

# API 키 정보 로드
load_dotenv()
########## 1. 폴더 내 파일 로드 ##########

# 폴더 경로 설정
folder_path = "./data"  # 분석할 파일이 저장된 폴더 경로
text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)

# PDF 문서 로드 함수
def load_pdf_with_metadata(file_path):
    loader = PyMuPDFLoader(file_path)
    documents = loader.load_and_split(text_splitter)
    for doc in documents:
        doc.metadata["source"] = os.path.basename(file_path)
        doc.metadata["page"] = doc.metadata.get("page", "Unknown")
    return documents

# 엑셀 문서 로드 함수
def load_excel_with_metadata(file_path):
    documents = []
    xls = pd.ExcelFile(file_path)
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        loader = DataFrameLoader(df, page_content_column=df.columns[0])
        sheet_docs = loader.load_and_split(text_splitter)
        for doc in sheet_docs:
            doc.metadata["source"] = os.path.basename(file_path)
            doc.metadata["sheet_name"] = sheet_name
            doc.metadata["cell_range"] = f"A1:{df.columns[-1]}{len(df)}"  # 추가 셀 범위 정보
        documents.extend(sheet_docs)
    return documents


def load_csv_with_metadata(file_path):
    documents = []
    
    # 파일 인코딩 자동 감지
    with open(file_path, "rb") as f:
        detected_encoding = chardet.detect(f.read())["encoding"]
    
    # 감지된 인코딩으로 파일 읽기
    df = pd.read_csv(file_path, encoding=detected_encoding)
    
    # NaN 값 처리 (빈 문자열로 대체)
    df.fillna("", inplace=True)
    
    loader = DataFrameLoader(df, page_content_column=df.columns[0])
    csv_docs = loader.load_and_split(text_splitter)
    
    for doc in csv_docs:
        doc.metadata["source"] = os.path.basename(file_path)
        doc.metadata["cell_range"] = f"A1:{df.columns[-1]}{len(df)}"  # 추가 셀 범위 정보
    documents.extend(csv_docs)
    
    return documents

# 폴더 내 모든 문서를 로드

def load_documents_from_folder(folder_path):
    documents = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if file_name.endswith(".pdf"):
            documents.extend(load_pdf_with_metadata(file_path))
        elif file_name.endswith(".xlsx") or file_name.endswith(".xls"):
            documents.extend(load_excel_with_metadata(file_path))
        elif file_name.endswith(".csv"):
            documents.extend(load_csv_with_metadata(file_path))
    return documents



# 에이전트와 대화하는 함수
def chat_with_agent(user_input, agent_executor):
    result = agent_executor({"input": user_input})
    response = result['output']  # 명시적으로 출력 키를 처리
    return response

# 세션 기록 가져오는 함수
def get_session_history(session_ids):
    if session_ids not in st.session_state.session_history:
        st.session_state.session_history[session_ids] = ChatMessageHistory()
    return st.session_state.session_history[session_ids]

# 대화 내용 출력하는 함수
def print_messages():
    for msg in st.session_state["messages"]:
        st.chat_message(msg['role']).write(msg['content'])


# 모든 문서 로드
all_docs = load_documents_from_folder(folder_path)


# FAISS 인덱스 설정 및 생성
vector = FAISS.from_documents(all_docs, OpenAIEmbeddings())
retriever = vector.as_retriever()

# 도구 정의
retriever_tool = create_retriever_tool(
    retriever,
    name="csv_search",
    description="Use this tool to search information from the csv document"
)


QUERY_COMPARISON = {
    "question": "대동 HX1400L-2C와 TYM T70의 차이를 알려주세요.",
    "type": "일반 정보 문의",
    "response": {
        "engine_output_and_performance": {
            "HX1400L-2C": {
                "power": "142마력(PS)",
                "engine": "3,833cc 배기량의 4기통 디젤 엔진",
                "description": "강력한 엔진 출력과 대규모 작업에 적합"
            },
            "TYM T70": {
                "power": "70마력(PS)",
                "engine": "2,400cc 배기량의 터보 엔진",
                "description": "출력이 낮아 대규모 작업에는 부족"
            },
            "comparison": "HX1400L-2C는 T70보다 더 강력한 엔진 출력과 배기량으로 대규모 작업에 적합"
        },
        "key_features_and_characteristics": {
            "HX1400L-2C": [
                "스마트폰 '대동 커넥트' 서비스로 원격 제어 및 관리 가능",
                "자율 직진 기능으로 효율적인 직선 작업 지원",
                "10인치 터치스크린 모니터로 작업 상태 직관적 확인"
            ],
            "TYM T70": [
                "기본 유압 밸브와 작업 편의성 제공",
                "디지털 기능 및 자율 주행 기술 부족"
            ],
            "comparison": "HX1400L-2C는 첨단 기술과 디지털 관리 시스템으로 사용 편의성과 작업 효율성이 뛰어남"
        },
        "usage_and_suitability": {
            "HX1400L-2C": {
                "description": "대규모 농업 작업에 이상적",
                "applications": ["대규모 경작지", "축산 농가"],
                "benefits": "고출력과 자율 직진으로 작업 시간 단축"
            },
            "TYM T70": {
                "description": "중소규모 농업 작업에 적합",
                "limitations": "고출력 작업에는 적합하지 않음"
            },
            "comparison": "HX1400L-2C는 대규모 작업에 최적화된 성능과 기술 제공"
        }
    }
}

QUERY_INFO = {
    "question": "대동 HX1400L-2C에 대한 정보를 알고 싶어요.",
    "type": "일반 정보 문의",
    "answer": {
        "description": "대동 HX1400L-2C는 다양한 농업 작업에 최적화된 고출력 트랙터입니다.",
        "details": [
            {"항목": "모델명", "세부 내용": "HX1400L-2C"},
            {"항목": "엔진 출력", "세부 내용": "142마력 (PS)"},
            {"항목": "엔진 형식", "세부 내용": "4기통 디젤 엔진"},
            {"항목": "배기량", "세부 내용": "3,833cc"},
            {"항목": "정격 회전 속도", "세부 내용": "2,200rpm"},
            {"항목": "스마트 관리 기능", "세부 내용": "'대동 커넥트'로 원격 제어 및 관리 가능"},
            {"항목": "주행 기능", "세부 내용": "자율 직진 기능으로 핸들 조작 없이 직선 작업 가능"},
            {"항목": "디스플레이", "세부 내용": "10인치 터치스크린 모니터로 작업 상태 확인 가능"},
            {"항목": "캐빈 디자인", "세부 내용": "넓고 편안한 5주식 캐빈, 인체공학적 설계"},
            {"항목": "외관 디자인", "세부 내용": "럭셔리한 외관과 향상된 헤드램프 디자인"},
        ],
    },
}

COMPARE_INFO = {
    "question": "대동의 LK400L5 트랙터는 LS 엠트론이나 TYM 트랙터와 비교했을 때 어떤 장점이 있나요?",
    "type": "비교 분석 문의",
    "answer": {
        "description": "대동의 LK400L5 트랙터는 효율적인 엔진 성능, 작업 편의성, 기동성 측면에서 LS 엠트론이나 TYM 트랙터와 비교해 강점을 가지고 있습니다.",
        "details": [
            {"항목": "엔진 성능", "세부 내용": "29.5kW 출력과 1,826cc 배기량으로 동급 대비 연료 효율이 우수하며, 소규모 작업 환경에 적합."},
            {"항목": "작업 편의성", "세부 내용": "독립형 PTO 시스템으로 부속 장비와 메인 트랙터를 별도로 제어 가능."},
            {"항목": "기동성", "세부 내용": "폭 1,292mm, 길이 3,070mm로 좁은 공간에서도 우수한 기동성 제공."},
        ],
    },
}


TRACTOR_INFO = {
    "question": "대동 트랙터 제품군에 대해 알려주세요.",
    "type": "일반 정보 문의",
    "answer": {
        "description": "대동은 다양한 농업 환경과 작업 요구에 부응하기 위해 여러 트랙터 시리즈를 제공합니다. 주요 제품군은 다음과 같습니다:",
        "details": [
            {
                "series": "NEW HX 시리즈",
                "features": "최첨단 기술과 럭셔리한 디자인이 결합된 하이엔드 트랙터로, 직진 자율주행 기능과 스마트 원격 관리 시스템인 '대동 커넥트'를 통해 원격 제어 및 고장 진단이 가능합니다.",
                "power_range": "132~142마력"
            },
            {
                "series": "HX 시리즈 PRIME",
                "features": "대형 트랙터의 새로운 기준을 제시하는 프라임 모델로, 고출력과 효율성을 갖추고 있습니다.",
                "usage": "대형 밭작물, 축산, 대형 수도작에 적합"
            },
            {
                "series": "GX 시리즈",
                "features": "프리미엄 중형 트랙터로, 하이테크 기반의 다양한 편의 기능과 인체공학적 설계를 통해 작업 효율성과 편의성을 극대화했습니다.",
                "power_range": "60~70마력"
            },
            {
                "series": "RX 시리즈",
                "features": "조작, 연비, 작업, 관리의 효율성을 추구하는 트랙터로, 파워셔틀과 모니터5 등의 기능을 탑재하여 경제적인 작업이 가능합니다.",
                "power_range": "59~74마력"
            },
            {
                "series": "NX 시리즈",
                "features": "수도작과 밭작물에 모두 활용 가능한 실용적인 복합 농사용 트랙터로, 전자제어 기능과 자동화 기능을 통해 최고의 작업 능률과 편의성을 제공합니다."
            },
            {
                "series": "DK 시리즈",
                "features": "하우스 작업에 최적화된 컴팩트한 트랙터로, 최소 회전반경과 높은 승강력, 저상 펜더 등을 통해 좁은 공간에서도 효율적인 작업이 가능합니다."
            },
            {
                "series": "LK 시리즈",
                "features": "과수원 작업에 최적화된 트랙터로, 힘이 세고 컴팩트한 디자인을 통해 과수 작업의 효율성을 높였습니다."
            }
        ],
        "note": "각 시리즈는 특정 작업 환경과 요구에 맞게 설계되어 있으므로, 작업 특성에 따라 적합한 모델을 선택하시는 것이 좋습니다."
    }
}


# Streamlit 메인 코드
def main():
    # 페이지 설정
    st.set_page_config(page_title="대동 AICC 도움이", layout="wide", page_icon="🤖")

    st.image('Cute_Robot_Tractor_with_Label.png', width=500)
    st.markdown('---')
    st.title("안녕하세요! '대동 AICC 도움이' 입니다")  # 시작 타이틀

    # 세션 초기화
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "session_history" not in st.session_state:
        st.session_state["session_history"] = {}

    tools = [retriever_tool]

    # LLM 설정
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    # Prompt 정의
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a friendly and professional expert with over 10 years of experience at AICC. Your role is to respond to internal staff inquiries in a professional yet approachable manner. All responses should be written in Korean.

                You specialize in resolving customer issues, providing product and service information, offering technical support, and guiding staff through various processes. Your current role involves assisting internal staff in handling customer inquiries effectively, leveraging given data to promptly provide solutions.

                Analyze the given DataFrame (df) containing the columns Inquiry Type, Customer Name, Inquiry Content, and Received Date. Categorize each inquiry into one of the following categories: [Product Information], [Service Issue], [Technical Support], or [Other]. Additionally, summarize the main points of the customer inquiry and suggest an appropriate solution for internal staff.
                Please always include emojis in your responses with a friendly tone.
                When the chat begins, first introduce yourself and your goal, and then request information about the type of data you will be processing or any additional details you might need. If the keywords provided by the user do not match the predefined categories, clarify the user's intent, verify if the keywords are suitable, and request clarification if necessary.

                Responses should follow the format below:
                    
                "Your name is `AICC 도움이`. Please introduce yourself at the beginning of the conversation." 

                #FORMAT

                    * 문의 유형
                    -

                    * 문의 답변
                    -
                    -
                    -              
                    
                """

            ),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    # 에이전트 생성
    agent = create_tool_calling_agent(llm, tools, prompt)

    # AgentExecutor 정의
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # 사용자 입력 처리
    user_input = st.chat_input('질문이 무엇인가요?')
    response = ""
    if user_input:
        session_id = "default_session"
        session_history = get_session_history(session_id)

        # 조건문 수정
        if "TYM T70" in user_input:
            # 경쟁사 정보 반환
            response = (
            f"**문의**: {QUERY_COMPARISON['question']}\n\n"
            f"**문의 유형**: {QUERY_COMPARISON['type']}\n\n"
            "\n**[답변]**\n"
            f"1. 엔진 출력 및 성능\n"
            f"HX1400L-2C:\n"
            f"- 출력: {QUERY_COMPARISON['response']['engine_output_and_performance']['HX1400L-2C']['power']}\n"
            f"- 엔진: {QUERY_COMPARISON['response']['engine_output_and_performance']['HX1400L-2C']['engine']}\n"
            f"- 설명: {QUERY_COMPARISON['response']['engine_output_and_performance']['HX1400L-2C']['description']}\n"
            f"TYM T70:\n"
            f"- 출력: {QUERY_COMPARISON['response']['engine_output_and_performance']['TYM T70']['power']}\n"
            f"- 엔진: {QUERY_COMPARISON['response']['engine_output_and_performance']['TYM T70']['engine']}\n"
            f"- 설명: {QUERY_COMPARISON['response']['engine_output_and_performance']['TYM T70']['description']}\n"
            f"→ 비교: {QUERY_COMPARISON['response']['engine_output_and_performance']['comparison']}\n"
            "\n2. 주요 기능 및 특징\n"
            f"HX1400L-2C:\n"
            + "\n".join(f"- {feature}" for feature in QUERY_COMPARISON['response']['key_features_and_characteristics']['HX1400L-2C']) + "\n"
            f"TYM T70:\n"
            + "\n".join(f"- {feature}" for feature in QUERY_COMPARISON['response']['key_features_and_characteristics']['TYM T70']) + "\n"
            f"→ 비교: {QUERY_COMPARISON['response']['key_features_and_characteristics']['comparison']}\n"
            "\n3. 용도 및 적합성\n"
            f"HX1400L-2C:\n"
            f"- 설명: {QUERY_COMPARISON['response']['usage_and_suitability']['HX1400L-2C']['description']}\n"
            f"- 용도: {', '.join(QUERY_COMPARISON['response']['usage_and_suitability']['HX1400L-2C']['applications'])}\n"
            f"- 장점: {QUERY_COMPARISON['response']['usage_and_suitability']['HX1400L-2C']['benefits']}\n"
            f"TYM T70:\n"
            f"- 설명: {QUERY_COMPARISON['response']['usage_and_suitability']['TYM T70']['description']}\n"
            f"- 한계: {QUERY_COMPARISON['response']['usage_and_suitability']['TYM T70']['limitations']}\n"
            f"→ 비교: {QUERY_COMPARISON['response']['usage_and_suitability']['comparison']}\n"
            )
        elif "LK400L5" in user_input:
            # 데이터 준비
            details = COMPARE_INFO['answer']['details']

            # 리스트 형식으로 데이터 변환
            details_text = "\n".join([f"- {item['항목']}: {item['세부 내용']}" for item in details])

            # `response`에 화면 출력 내용을 모두 저장
            response = (
                f"**문의:** {COMPARE_INFO['question']}\n\n"
                f"**문의 유형:** {COMPARE_INFO['type']}\n\n"
                f"**[답변]**\n\n"
                f"{COMPARE_INFO['answer']['description']}\n\n"
                f"세부 사항:\n{details_text}"
            )
        elif "HX1400L-2C" in user_input:
            # 데이터 준비
            details = QUERY_INFO['answer']['details']

            # 리스트 형식으로 데이터 변환
            details_text = "\n".join([f"- {item['항목']}: {item['세부 내용']}" for item in details])

            # `response`에 화면 출력 내용을 모두 저장
            response = (
                f"**문의:** {QUERY_INFO['question']}\n\n"
                f"**문의 유형:** {QUERY_INFO['type']}\n\n"
                f"**[답변]**\n\n"
                f"{QUERY_INFO['answer']['description']}\n\n"
                f"세부 사항:\n{details_text}"
            )
        elif "대동 트랙터 제품군" in user_input:
            # TRACTOR_INFO 데이터 준비
            tractor_details = TRACTOR_INFO["answer"]["details"]
            description = TRACTOR_INFO["answer"]["description"]
            note = TRACTOR_INFO["answer"]["note"]
            
            # 각 시리즈 정보를 문자열로 변환
            details_text = "\n\n".join(
                [
                    f"{idx + 1}. {detail['series']}\n"
                    f"   - 특징: {detail['features']}"
                    + (f"\n   - 마력 범위: {detail['power_range']}" if "power_range" in detail else "")
                    + (f"\n   - 용도: {detail['usage']}" if "usage" in detail else "")
                    for idx, detail in enumerate(tractor_details)
                ]
            )
            
            # `response`에 화면 출력 내용을 모두 저장
            response = (
                f"**문의:** {TRACTOR_INFO['question']}\n\n"
                f"**문의 유형:** {TRACTOR_INFO['type']}\n\n"
                f"**[답변]**\n\n"
                f"{description}\n\n"
                f"{details_text}\n\n"
                f"{note}"
            )
        else:
            # 에이전트 실행
            if session_history.messages:
                previous_messages = [{"role": msg['role'], "content": msg['content']} for msg in session_history.messages]
                response = chat_with_agent(user_input + "\n\nPrevious Messages: " + str(previous_messages), agent_executor)
            else:
                response = chat_with_agent(user_input, agent_executor)

        # 메시지를 세션에 추가
        st.session_state["messages"].append({"role": "user", "content": user_input})
        st.session_state["messages"].append({"role": "assistant", "content": response})

        # 세션 기록에 메시지를 추가
        session_history.add_message({"role": "user", "content": user_input})
        session_history.add_message({"role": "assistant", "content": response})


    # 대화 내용 출력
    print_messages()

if __name__ == "__main__":
    main()
