from typing import List, Optional
from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser

model = ChatOpenAI(
    model="ChatCoder",
    base_url="http://127.0.0.1:8000/v1",
    api_key="sk-abc123"# 请替换成你的 API key
)


class Person(BaseModel):
    name: str = Field(description="人物姓名")
    age: Optional[int] = Field(description="人物年龄，可缺失")

class Information(BaseModel):
    people: List[Person] = Field(description="人物信息列表")

info_fn = convert_to_openai_function(Information)
extraction_model = model.bind(functions=[info_fn], function_call={"name": "Information"})

prompt = ChatPromptTemplate.from_messages([
    ("system", "提取文本中所有人物姓名和年龄"),
    ("user", "{input}")
])

extraction_chain = prompt | extraction_model | JsonOutputFunctionsParser()

print(extraction_chain.invoke({"input": "Joe is 30, his mom is Martha"}))
