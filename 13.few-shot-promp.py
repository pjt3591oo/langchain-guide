from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import FewShotPromptTemplate

# 예제 템플릿 정의
example_template = """
### 질문 ###
{question}

### 답변 ###
{answer}
"""
example_prompt = PromptTemplate.from_template(example_template)

# 기술 관련 예제 데이터로 변경
examples = [
    ("클라우드 컴퓨팅이란 무엇인가요?", 
     "클라우드 컴퓨팅은 인터넷을 통해 서버, 스토리지, 데이터베이스, 네트워킹, 소프트웨어 등의 컴퓨팅 리소스를 필요에 따라 제공하는 서비스입니다."),
    
    ("머신러닝과 딥러닝의 차이점은 무엇인가요?", 
     "머신러닝은 데이터로부터 패턴을 학습하는 AI의 한 분야이고, 딥러닝은 신경망을 여러 층으로 쌓아 복잡한 패턴을 학습하는 머신러닝의 하위 분야입니다."),
    
    ("블록체인 기술의 주요 특징을 설명해주세요.", 
     "블록체인의 주요 특징은 분산 원장, 암호화, 불변성, 투명성, 탈중앙화입니다. 이를 통해 신뢰할 수 있는 제3자 없이도 안전한 거래가 가능합니다."),
    
    ("5G 기술이 4G와 비교하여 가지는 장점은 무엇인가요?", 
     "5G는 4G보다 더 빠른 데이터 전송 속도, 낮은 지연 시간, 더 많은 기기 연결 가능, 향상된 에너지 효율성, 그리고 더 높은 신뢰성을 제공합니다."),
    
    ("컨테이너화(Containerization)란 무엇인가요?", 
     "컨테이너화는 애플리케이션과 그 종속성을 하나의 패키지(컨테이너)로 묶어 어떤 환경에서도 일관되게 실행될 수 있도록 하는 기술입니다. Docker가 대표적인 컨테이너 플랫폼입니다."),
    
    ("양자 컴퓨팅의 기본 원리는 무엇인가요?", 
     "양자 컴퓨팅은 양자역학의 원리를 활용하여 연산을 수행합니다. 양자 비트(큐비트)는 0과 1의 상태를 동시에 가질 수 있는 중첩 상태를 활용하여 기존 컴퓨터보다 특정 문제를 훨씬 빠르게 해결할 수 있습니다.")
]

# 예제를 딕셔너리 형태로 변환하는 함수
def format_examples(examples):
    return [{"question": q, "answer": a} for q, a in examples]

# FewShotPromptTemplate 생성
prompt = FewShotPromptTemplate(
    examples=format_examples(examples),  # 함수를 통해 예제 변환
    example_prompt=example_prompt,       # 예제 포맷팅 템플릿
    prefix="다음은 질문과 답변의 예시입니다:\n",  # 접두사 추가
    suffix="\n### 질문 ###\n{input}\n\n### 답변 ###",  # 기술 관련 질문에 대한 접미사
    input_variables=["input"],           # 입력 변수
    example_separator="\n\n"             # 예제 사이의 구분자
)

# 새로운 질문에 대한 프롬프트 생성 및 출력
print(prompt.invoke({"input": "인공지능의 윤리적 고려사항에는 어떤 것들이 있나요?"}).to_string())