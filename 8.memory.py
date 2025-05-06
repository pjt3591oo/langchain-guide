from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory import ConversationSummaryMemory
from langchain.memory import ConversationSummaryBufferMemory
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()

chat_model = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    temperature=0.7, 
    model_name="l3-8b-stheno-v3.1-iq-imatrix"
)

# 1. ConversationBufferMemory
buffer_memory = ConversationBufferMemory()
buffer_memory.save_context({"input": "안녕하세요!"}, {"output": "반갑습니다! 무엇을 도와드릴까요?"})
print(buffer_memory.load_memory_variables({}))
# 출력: {'history': 'Human: 안녕하세요!\nAI: 반갑습니다! 무엇을 도와드릴까요?'}

# 2. ConversationBufferWindowMemory
window_memory = ConversationBufferWindowMemory(k=2)
window_memory.save_context({"input": "안녕하세요!"}, {"output": "반갑습니다! 무엇을 도와드릴까요?"})
print(window_memory.load_memory_variables({}))
# 출력: {'history': 'Human: 안녕하세요!\nAI: 반갑습니다! 무엇을 도와드릴까요?'}

# 3. ConversationSummaryMemory
summary_memory = ConversationSummaryMemory(llm=chat_model)
summary_memory.save_context({"input": "안녕하세요!"}, {"output": "반갑습니다! 무엇을 도와드릴까요?"})
print(summary_memory.load_memory_variables({}))
# 출력: {'history': 'Human: 안녕하세요!\nAI: 반갑습니다! 무엇을 도와드릴까요?'}

