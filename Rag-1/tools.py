from langchain.tools import tool
from indexing import vectorstore


last_retrieved_context=[]

@tool
def top_k_context(query:str):
    """retrieve top k elemnets from your vector store"""
    global last_retrieved_context
    
    docs = vectorstore.similarity_search(query, k=3)

    last_retrieved_context = [doc.page_content for doc in docs]


    return "\n\n".join(last_retrieved_context)
