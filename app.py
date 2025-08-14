import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler

## Set up thr streamlit App
st.set_page_config(page_title="Text to Math Problem Solver And Data Search Assistant")
st.title("Text To Math Problem Solver Using Gemma Model")

groq_api_key = st.sidebar.text_input(label="Groq API KEY",type="password")

if not groq_api_key:
    st.info("Please add your Groq API Key to continue")
    st.stop()

llm = ChatGroq(model="Gemma2-9b-It",groq_api_key=groq_api_key)


## Initialize the tools
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="A tool for searching the Internet to find the various information on the topics mentioned."
)


## Initialize the Math Tool

math_chain = LLMMathChain.from_llm(llm=llm)
calculator=Tool(
    name="Calculator",
    func=math_chain.run,
    description = "A tool for answering the math related questions. only input mathematical expression need to be provided"
)

prompt = """
You are a agent tasked for solving users mathematical questions. Logically arrive at the solution and provide a detailed explaination
and display it pointwise for the question below
Question:{question}
Answer:
"""

prompt_template = PromptTemplate(
    input_variables=['question'],
    template=prompt
)

## Combine all the tools into chain
chain = LLMChain(llm=llm,prompt=prompt_template)

reasoning_tool = Tool(
    name="Reasoning tool",
    func=chain.run,
    description="A tol for answering logic based and reasoning questions."
)

## Initialize the agents
assistant_agent = initialize_agent(
    tools=[wikipedia_tool,calculator, reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

if "messages" not in st.session_state:
    st.session_state["messages"]=[
         {"role":"assistant","content":"Hi, I am a Math Chatbot who can answer all your maths question."}
    ]


for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])


## Function to generate the response
# def genearte_response(question):
#     response=assistant_agent.invoke({'input':question})
#     return response

## Lets start the interaction
question = st.text_area("Enter your question:","Three boxes are labeled 'apples,' 'oranges,' and 'apples and oranges,' but all labels are wrong—by picking one fruit from one box, how can you correctly identify the contents of all three boxes?")


if st.button("Find my answer"):
    if question:
        with st.spinner("Generate Response.."):
            st.session_state.messages.append({"role":"user","content":question})
            st.chat_message("user").write(question)

            st_cb = StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
            response=assistant_agent.run(st.session_state.messages,callbacks=[st_cb])
            st.session_state.messages.append({'role':'assistant','content':response})
            st.write("###Response")
            st.success(response)
    else:
       st.warning("Please enter the question")
