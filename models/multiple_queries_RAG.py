from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

from tools import prompt_templates_retrieve
from tools.invoke_result import invoke_generate_queries_with_origin


def multiple_queries_chain(model):
    
    prompt_multi_query = PromptTemplate(
        template=prompt_templates_retrieve.prompt_multi_query,
        input_variables=["question", "number_questions"],
    )
   
    generate_queries_chain = (
        {
            "question": itemgetter("question"),
            "question_numbers": itemgetter("question_numbers"),
        }
        | prompt_multi_query
        | model
        | StrOutputParser()
    )
    generate_queries_chain = (
        {"question": itemgetter("question"), "alternatives": generate_queries_chain}
        | RunnableLambda(invoke_generate_queries_with_origin)
        | (lambda x: x.split("\n"))
    )

    return generate_queries_chain