from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import FewShotPromptTemplate
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
# 기술 관련 예제 데이터 준비
examples = [
    {
        "question": "클라우드 컴퓨팅이란 무엇인가요?",
        "answer": "클라우드 컴퓨팅은 인터넷을 통해 서버, 스토리지, 데이터베이스, 네트워킹, 소프트웨어 등의 컴퓨팅 리소스를 필요에 따라 제공하는 서비스입니다."
    },
    {
        "question": "머신러닝과 딥러닝의 차이점은 무엇인가요?",
        "answer": "머신러닝은 데이터로부터 패턴을 학습하는 AI의 한 분야이고, 딥러닝은 신경망을 여러 층으로 쌓아 복잡한 패턴을 학습하는 머신러닝의 하위 분야입니다."
    },
    {
        "question": "블록체인 기술의 주요 특징을 설명해주세요.",
        "answer": "블록체인의 주요 특징은 분산 원장, 암호화, 불변성, 투명성, 탈중앙화입니다. 이를 통해 신뢰할 수 있는 제3자 없이도 안전한 거래가 가능합니다."
    },
    {
        "question": "5G 기술이 4G와 비교하여 가지는 장점은 무엇인가요?",
        "answer": "5G는 4G보다 더 빠른 데이터 전송 속도, 낮은 지연 시간, 더 많은 기기 연결 가능, 향상된 에너지 효율성, 그리고 더 높은 신뢰성을 제공합니다."
    },
    {
        "question": "컨테이너화(Containerization)란 무엇인가요?",
        "answer": "컨테이너화는 애플리케이션과 그 종속성을 하나의 패키지(컨테이너)로 묶어 어떤 환경에서도 일관되게 실행될 수 있도록 하는 기술입니다. Docker가 대표적인 컨테이너 플랫폼입니다."
    },
    {
        "question": "양자 컴퓨팅의 기본 원리는 무엇인가요?",
        "answer": "양자 컴퓨팅은 양자역학의 원리를 활용하여 연산을 수행합니다. 양자 비트(큐비트)는 0과 1의 상태를 동시에 가질 수 있는 중첩 상태를 활용하여 기존 컴퓨터보다 특정 문제를 훨씬 빠르게 해결할 수 있습니다."
    },
    {
        "question": "API란 무엇이며 어떤 역할을 하나요?",
        "answer": "API(Application Programming Interface)는 서로 다른 소프트웨어 간의 상호작용을 가능하게 하는 인터페이스입니다. 개발자가 특정 기능을 쉽게 구현할 수 있도록 하며, 시스템 간의 통합을 용이하게 합니다."
    },
    {
        "question": "마이크로서비스 아키텍처의 장점은 무엇인가요?",
        "answer": "마이크로서비스 아키텍처는 독립적으로 배포 가능한 작은 서비스로 구성되어 있어 개발 속도 향상, 확장성 개선, 장애 격리, 기술 다양성 허용, 그리고 팀 자율성 증대 등의 장점이 있습니다."
    }
]

# 예제 템플릿 정의
example_prompt = PromptTemplate.from_template("질문: {question}\n답변: {answer}")

# SemanticSimilarityExampleSelector 설정
# 참고: 실제 사용 시 OpenAI API 키가 필요합니다
model_name = "sentence-transformers/all-MiniLM-L6-v2" # 영어권에서 성능 좋은 경량 모델
# model_name = "jhgan/ko-sroberta-multitask" # 한국어 모델 예시
model_kwargs = {'device': 'cpu'} # CPU 사용 명시 (GPU 사용 가능 시 'cuda')
encode_kwargs = {'normalize_embeddings': False} # 정규화 여부

embeddings_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

vector_store = Chroma(persist_directory="./chroma_db_example", embedding_function=embeddings_model)

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,                      # 예제 데이터
    embeddings_model,                    # 임베딩 모델
    vector_store,
    k=2,                           # 선택할 예제 수
)

# FewShotPromptTemplate 생성
prompt = FewShotPromptTemplate(
    example_selector=example_selector,  # 예제 선택기
    example_prompt=example_prompt,      # 예제 포맷팅 템플릿
    prefix="다음은 기술 관련 질문과 답변의 예시입니다. 이 예시를 참고하여 질문에 답변해주세요:\n\n",
    suffix="\n\n질문: {input}\n답변:",
    input_variables=["input"]
)

# 실행 예시
def get_response(question):
    # 실제 환경에서는 이 부분에 LLM 호출 코드가 들어갑니다
    formatted_prompt = prompt.invoke({"input": question}).to_string()
    print("생성된 프롬프트:")
    print(formatted_prompt)
    return formatted_prompt

# 예시 실행
print("--------------1------------------")
question = "인공지능 모델을 훈련할 때 과적합 문제를 어떻게 해결할 수 있나요?"
get_response(question)

print("--------------2------------------")
# 다른 예시
question2 = "클라우드 네이티브 애플리케이션이란 무엇인가요?"
get_response(question2)