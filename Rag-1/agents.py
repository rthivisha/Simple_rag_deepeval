import tools as t1
from prompts import rag_prompt
from objects import answer
from langchain.messages import HumanMessage

from tools import top_k_context

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI


llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.2)

agent = create_agent(model = llm,tools=[top_k_context],system_prompt = rag_prompt,response_format=answer)


def ask(query:str) ->dict:
    t1.last_retrieved_context =[]

    result = agent.invoke({"messages":[HumanMessage(content=query)]})

    structured_response=result["structured_response"]

    return {
        "ans":structured_response.ans,
        "context":structured_response.context,
    }

if __name__ == "__main__":
    print(ask("what are the rag evaluation metrics used?"))

