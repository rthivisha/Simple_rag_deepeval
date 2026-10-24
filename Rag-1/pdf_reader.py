from pymupdf import open 
import re
from typing import List


paragraphs = []

def extract_pdf(path_of_pdf:str) -> List:

    doc = open (path_of_pdf)

    for page in doc:
        blocks = sorted(page.get_text("blocks"),key=lambda b:(b[1],b[0]))

        for block in blocks:
            text = block[4]
            text=re.sub(r"\r+"," ",text)

            paragraphs.append(text)



    return paragraphs

if __name__ == "__main__":
    print(len(extract_pdf("pdfs/llm_agent_eval_metrics_text.pdf")))
