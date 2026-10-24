from pydantic import BaseModel ,  Field


class answer(BaseModel):
    ans:str = Field(description="Answer from the context from vectorstore")
    context:str = Field(description="Provide relevant context from the pdf that you used to answer the question")
