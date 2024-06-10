from langchain_openai.chat_models import ChatOpenAI
from langchain.output_parsers import JsonOutputKeyToolsParser
from langchain_core.tools import tool


@tool
def multiply(first_int: int, second_int: int) -> int:
    """Multiply two integers together."""
    return first_int * second_int


model = ChatOpenAI(base_url="http://localhost:1234/v1",
                   api_key="not-needed", temperature=0.7, streaming=True, model="local-model")

model_with_tools = model.bind_tools([multiply], tool_choice="multiply")
model_with_tools.kwargs["tools"]
[{'type': 'function',
  'function': {'name': 'multiply',
               'description': 'multiply(first_int: int, second_int: int) -> int - Multiply two integers together.',
               'parameters': {'type': 'object',
                              'properties': {'first_int': {'type': 'integer'},
                                             'second_int': {'type': 'integer'}},
                              'required': ['first_int', 'second_int']}}}]

model_with_tools.kwargs["tool_choice"]
{'type': 'function', 'function': {'name': 'multiply'}}

# Note: the `.map()` at the end of `multiply` allows us to pass in a list of `multiply` arguments instead of a single one.
chain = (
    model_with_tools
    | JsonOutputKeyToolsParser(key_name="multiply", return_single=True)
    | multiply
)
chain.invoke("What's four times 23")
print(chain.invoke("What's four times 23"))
