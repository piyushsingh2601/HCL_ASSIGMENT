import streamlit as st
from langchain.prompts import PromptTemplate
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline


# Load the HuggingFace pipeline with the clean_up_tokenization_spaces parameter set
model = pipeline("text2text-generation", model="google/flan-t5-large", clean_up_tokenization_spaces=True,max_new_tokens= 1000)
llm = HuggingFacePipeline(pipeline=model)

# Define the template
template = "Answer the following query with detailed reasoning: {query}"
prompt = PromptTemplate(input_variables=["query"], template=template)

# Create a RunnableSequence, combining the prompt and LLM
chain = prompt | llm

# Set the max token length for the response
max_length = 4000

# Streamlit app interface
st.title("Stateless LLM with LangChain")

# User input
query = st.text_input("Enter your query:", "")

if st.button("Submit"):
    if query:
        try:
            # Run the query through the chain and generate response
            response = chain.invoke({"query": query})
            st.write("Response:", response)
        except Exception as e:
            st.write(f"An error occurred: {e}")
    else:
        st.write("Please enter a query.")
