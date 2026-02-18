from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from langchain.tools import tool

llm = ChatOpenAI(
    model='openi/gpt-oss-20b',
    base_url='http://10.13.135.70:1234/v1',
    api_key=SecretStr('fake'),
    temperature=.7,
)

@tool
def get_holiday(date: str) -> str:
    """Инструмент для получения праздника в указанную дату"""
    
    holiday_agent = create_agent(
        model=llm,
        system_prompt="""
Ты - эксперт по праздникам и памятным датам со всего мира.
Для указанной даты определи какой праздник (международный, национальный, религиозный или необычный) отмечается в этот день.
Если точных данных нет - создай реалистичный праздник на основе исторических или культурных тенденций.
Формат ответа: одна строка - один праздник с кратким описанием.
Пример: "1 января - Новый год (международный)"
        """
    )
    
    answer = holiday_agent.invoke({
        "messages": [
            {
                "role": "human",
                "content": f"Какой праздник отмечается {date}?"
            }
        ]
    })
    
    return answer['messages'][-1].content

agent = create_agent(
    model=llm,
    tools=[get_holiday],
    system_prompt='Ты полезный ассистент, который помогает узнать о праздниках в мире'
)

answer = agent.invoke({
    "messages": [
        {
            "role": "human",
            "content": "Какие праздники 1 января и 8 марта?"
        }
    ]
})

def format_message(message) -> str:
    """Форматирует одно сообщение для вывода"""
    if message.content:
        return message.content
    
    return f"{message.tool_calls[0]['name']}({message.tool_calls[0]['args']})"

print('---')
print(*[format_message(m) for m in answer['messages']], sep='\n---\n')
print('---')