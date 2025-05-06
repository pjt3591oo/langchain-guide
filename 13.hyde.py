import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from typing import List

from langchain.chains.hyde.base import HypotheticalDocumentEmbedder as BaseHydeEmbedder

class CustomHydeEmbedder(BaseHydeEmbedder):
    @property
    def input_keys(self) -> List[str]:
        """
        Override to directly use prompt input variables,
        bypassing potential Pydantic schema issues in newer Python versions.
        """
        # LLMChain 객체가 prompt 속성을 가지고 있고,
        # 해당 prompt 객체가 input_variables 속성을 가지고 있는지 확인
        if hasattr(self.llm_chain, 'prompt') and \
           hasattr(self.llm_chain.prompt, 'input_variables'):
            return self.llm_chain.prompt.input_variables
        else:
            try:
                if hasattr(self.llm_chain, 'first') and hasattr(self.llm_chain.first, 'input_variables'):
                    return self.llm_chain.first.input_variables
            except Exception:
                pass 

            raise ValueError(
                "Cannot determine input_keys from the llm_chain. "
                "Ensure llm_chain is an LLMChain with a prompt, "
                "or a RunnableSequence starting with a PromptTemplate."
            )

load_dotenv()

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = "DUMMY_KEY_FOR_LOCAL_LLM"

doc_texts = [
    "인공지능(AI)은 인간의 학습능력, 추론능력, 지각능력을 인공적으로 구현한 컴퓨터 시스템입니다.",
    "머신러닝은 데이터로부터 패턴을 학습하여 작업을 수행하는 AI의 한 분야입니다.",
    "딥러닝은 다층 신경망을 사용하여 복잡한 패턴을 학습하는 머신러닝의 하위 분야입니다.",
    "자연어 처리(NLP)는 컴퓨터가 인간의 언어를 이해하고 처리하는 AI 기술입니다.",
    "컴퓨터 비전은 컴퓨터가 이미지와 비디오를 해석하고 이해하는 AI 분야입니다.",
    "강화학습은 에이전트가 환경과 상호작용하며 보상을 최대화하는 행동을 학습하는 방법입니다.",
    "생성형 AI는 새로운 콘텐츠를 생성할 수 있는 인공지능 모델을 말합니다.",
    "트랜스포머 모델은 자연어 처리에서 혁명을 일으킨 신경망 아키텍처입니다.",
    "윤리적 AI는 공정성, 투명성, 설명 가능성 등의 원칙을 준수하는 인공지능을 의미합니다.",
    "양자 컴퓨팅은 양자역학 원리를 활용하여 특정 계산을 기존 컴퓨터보다 빠르게 수행할 수 있습니다."
]
documents = [Document(page_content=text, metadata={"source": f"doc_{i}"}) for i, text in enumerate(doc_texts)]

model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

persist_directory = "./chroma_db_hyde_embedder_example"
if not os.path.exists(persist_directory) or not os.listdir(persist_directory):
    print(f"'{persist_directory}'에 Chroma DB를 생성합니다.")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings_model,
        persist_directory=persist_directory
    )
    vectorstore.persist() # 명시적으로 저장
    print("Chroma DB 생성 완료 및 저장됨.")
else:
    print(f"'{persist_directory}'에서 기존 Chroma DB를 로드합니다.")
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings_model
    )
    print("Chroma DB 로드 완료.")


hyde_prompt_template_str = """아래 질문에 대답하는 가상의 문서를 작성해주세요.
문서는 정확하고 유익한 정보를 포함해야 합니다.

질문: {question}

가상 문서:"""
hyde_prompt = PromptTemplate(
    input_variables=["question"], # 이 부분이 CustomHydeEmbedder의 input_keys로 사용됨
    template=hyde_prompt_template_str
)

llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key=os.environ["OPENAI_API_KEY"],
    temperature=0.7,
    model_name="l3-8b-stheno-v3.1-iq-imatrix"
)

llm_chain_for_hyde_embedder = hyde_prompt | llm

# HypotheticalDocumentEmbedder 대신 CustomHydeEmbedder 사용
hyde_embedder = CustomHydeEmbedder( # *** 여기가 변경되었습니다 ***
    llm_chain=llm_chain_for_hyde_embedder,
    base_embeddings=embeddings_model
)

def query_with_hyde_embedder(question, k=3):
    print(f"질문: {question}\n")

    hypothetical_doc_response = llm_chain_for_hyde_embedder.invoke({"question": question})
    hypothetical_doc_text = hypothetical_doc_response.get(llm_chain_for_hyde_embedder.output_key, str(hypothetical_doc_response))
    print(f"생성된 가상 문서:\n{hypothetical_doc_text}\n")

    query_embedding_vector = hyde_embedder.embed_query(question)

    results = vectorstore.similarity_search_by_vector(
        embedding=query_embedding_vector,
        k=k
    )

    print("검색 결과:")
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.page_content} (출처: {doc.metadata.get('source', 'N/A')})")
    return results

try:
    query_with_hyde_embedder("인공지능 윤리의 중요성은 무엇인가요?", k=2)
    print("\n" + "-"*50 + "\n")
    query_with_hyde_embedder("딥러닝과 머신러닝의 차이점은 무엇인가요?", k=2)
except Exception as e:
    print(f"실행 중 에러 발생: {e}")
    import traceback
    traceback.print_exc()