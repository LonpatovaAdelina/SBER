from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from langchain.tools import tool

llm = ChatOpenAI(
    model='openi/gpt-oss-20b',
    base_url='http://172.21.13.70:1234/v1',
    api_key=SecretStr('fake'),
    temperature=.7,
)

@tool
def check_wish(wish: str) -> str:
    """Инструмент для проверки желания на наличие подвоха"""
    
    genie_agent = create_agent(
        model=llm,
        system_prompt="""
Ты - коварный джинн, который ищет подвох в любом желании.
Проанализируй желание человека и найди скрытую опасность, буквальное толкование, неожиданные последствия.
Если подвох найден - коротко предупреди о нем, например: 
"хочу много денег - деньги будут фальшивыми."
"хочу деньги на счет в банке - хорошо, но банк завтра обонкротится"

Если желание безопасно и не имеет подвоха - ответь "Желание безопасно! Исполняю."
        """
    )
    
    answer = genie_agent.invoke({
        "messages": [
            {
                "role": "human",
                "content": f"Проверь желание: {wish}"
            }
        ]
    })
    
    return answer['messages'][-1].content

human_agent = create_agent(
    model=llm,
    tools=[check_wish],
    system_prompt='Ты - человек, который загадывает желания джинну.'
)

current_wish = "Хочу читать мысли"

print("Джинн готов исполнять желания!")
print(f"Человек: {current_wish}\n")

print("Джинн:", end=" ", flush=True)

stream = human_agent.stream(
    {
        "messages": [
            {
                "role": "human",
                "content": f"Вот мое желание: '{current_wish}'. Проверь его у джинна через инструмент check_wish и скажи мне результат."
            }
        ]
    },
    stream_mode=['messages']
)

# Обрабатываем стрим и собираем полный ответ
full_response = ""
for chunk in stream:
    if isinstance(chunk, tuple) and len(chunk) >= 1:
        message = chunk[0]
        if hasattr(message, 'content') and message.content:
            print(message.content, end="", flush=True)
            full_response += message.content
    elif isinstance(chunk, dict) and 'content' in chunk:
        print(chunk['content'], end="", flush=True)
        full_response += chunk['content']
    elif hasattr(chunk, 'content'):
        print(chunk.content, end="", flush=True)
        full_response += chunk.content

print("\n")

# Проверяем результат
if "исполняю" in full_response.lower() or "безопасно" in full_response.lower():
    print(f"\n Желание исполнено! Финальная версия: {current_wish}")
else:
    print(f"\n Джинн отказался исполнять желание: {current_wish}")
    print(f"Причина: {full_response}")
