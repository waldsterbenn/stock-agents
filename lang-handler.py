from langchain_core.messages import BaseMessage
import operator
from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langchain_core.messages import FunctionMessage
import json
from langgraph.prebuilt import ToolInvocation
from langchain.tools import tool
from langgraph.prebuilt import ToolExecutor
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_function
from langgraph.prebuilt import create_agent_executor
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain_core.messages import AIMessage
import json
import os
from uuid import uuid4


from fu_stock_analyzer_tool import hum_stock_analyzer_tool

unique_id = uuid4().hex[0:8]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"Tracing Walkthrough - {unique_id}"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
# Update to your API key
os.environ["LANGCHAIN_API_KEY"] = "ls__2b5801d8faec43858bf1436e9c23e0d3"

# Used by the agent in this tutorial
# os.environ["OPENAI_API_KEY"] = "<YOUR-OPENAI-API-KEY>"


@tool("stock_analyzer_tool", return_direct=True)
def stock_analyzer_tool(ticker_symbol: str) -> str:
    """name: Returns the name of the pickled pandas dataframe, with the results of the techical indicators."""
    print("Calling function")
    return hum_stock_analyzer_tool(ticker_symbol)


tools = [stock_analyzer_tool]
tool_executor = ToolExecutor(tools)

# Point to the local server
model = ChatOpenAI(base_url="http://localhost:1234/v1",
                   api_key="not-needed", temperature=0.7, streaming=True, model="local-model")

history = [
    {"role": "system", "content": "You are an intelligent assistant. You always provide well-reasoned answers that are both correct and helpful."},
]


functions = [convert_to_openai_function(t) for t in tools]
model = model.bind_functions(functions)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


# Define the function that determines whether to continue or not

def should_continue(state):
    messages = state['messages']
    last_message = messages[-1]
    # If there is no function call, then we finish
    if "function_call" not in last_message.additional_kwargs:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"

# Define the function that calls the model


def call_model(state):
    messages = state['messages']
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

# Define the function to execute tools


def call_tool(state):
    messages = state['messages']
    # Based on the continue condition
    # we know the last message involves a function call
    last_message = messages[-1]
    # We construct an ToolInvocation from the function_call
    action = ToolInvocation(
        tool=last_message.additional_kwargs["function_call"]["name"],
        tool_input=json.loads(
            last_message.additional_kwargs["function_call"]["arguments"]),
    )
    # We call the tool_executor and get back a response
    response = tool_executor.invoke(action)
    # We use the response to create a FunctionMessage
    function_message = FunctionMessage(content=str(response), name=action.tool)
    # We return a list, because this will get added to the existing list
    return {"messages": [function_message]}


def first_model(state):
    human_input = state["messages"][-1].content
    return {
        "messages": [
            AIMessage(
                content="",
                additional_kwargs={
                    "function_call": {
                        "name": "stock_analyzer_tool",
                        "arguments": json.dumps({"query": human_input}),
                    }
                },
            )
        ]
    }


# Define a new graph
workflow = StateGraph(AgentState)

# Define the new entrypoint
workflow.add_node("first_agent", first_model)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.set_entry_point("first_agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
    # Finally we pass in a mapping.
    # The keys are strings, and the values are other nodes.
    # END is a special node marking that the graph should finish.
    # What will happen is we will call `should_continue`, and then the output of that
    # will be matched against the keys in this mapping.
    # Based on which one it matches, that node will then be called.
    {
        # If `tools`, then we call the tool node.
        "continue": "action",
        # Otherwise we finish.
        "end": END
    }
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge('action', 'agent')

# After we call the first agent, we know we want to go to action
workflow.add_edge("first_agent", "action")

# This compiles it into a LangChain Runnable, meaning you can use it as you would any other runnable
app = workflow.compile()

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/openai-functions-agent")
# Construct the OpenAI Functions agent
agent_runnable = create_openai_functions_agent(model, tools, prompt)

app = create_agent_executor(agent_runnable, tools)

inputs = {"input": "Your job is to analyse the data table and reccommend whether the stock is a Buy or Sell. Create a short report in bullitpoints that describes why. The stock ticker is: AAPL",
          "chat_history": []}

for output in app.stream(inputs):
    # stream() yields dictionaries with output keyed by node name
    for key, value in output.items():
        print(f"Output from node '{key}':")
        print("---")
        print(value)
    print("\n---\n")
