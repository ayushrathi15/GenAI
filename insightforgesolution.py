import streamlit as st
import pandas as pd
import plotly.express as px
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# --- 1. Configuration & Setup ---
st.set_page_config(page_title="InsightForge BI Assistant", layout="wide")

# Initialize LLM (Ensure your OPENAI_API_KEY is set in your environment)
llm = ChatOpenAI(model="gpt-4", temperature=0)

# --- 2. RAG Logic (Structured Data Retrieval) ---
def get_data_context(df):
    """
    Acts as the 'Retriever'. Instead of vectors, we retrieve 
    statistical metadata to feed the LLM.
    """
    stats = df.describe().to_string()
    columns = ", ".join(df.columns.tolist())
    sample = df.head(3).to_string()
    return f"Columns: {columns}\n\nDescriptive Stats:\n{stats}\n\nData Sample:\n{sample}"

# --- 3. Prompt Engineering ---
template = """
You are InsightForge, an expert Business Intelligence Assistant. 
Use the provided data context to answer the user's business question accurately.

Data Context:
{context}

Chat History:
{chat_history}

User Question: {question}

Instructions:
- If the user asks for a calculation, use the descriptive stats provided.
- If the data doesn't contain the answer, be honest and say so.
- Provide actionable recommendations based on the trends you see.

Answer:"""

prompt = PromptTemplate(
    input_variables=["context", "chat_history", "question"], 
    template=template
)

# --- 4. Memory Integration ---
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history")

bi_chain = LLMChain(llm=llm, prompt=prompt, memory=st.session_state.memory)

# --- 5. Streamlit User Interface ---
st.title("🛠️ InsightForge: AI Business Intelligence")
st.markdown("Transforming raw data into actionable insights using Structured RAG.")

# File Upload
uploaded_file = st.sidebar.file_uploader("Upload your Business CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Create Tabs for different views
    tab1, tab2 = st.tabs(["💬 AI Assistant", "📊 Visual Dashboard"])

    with tab1:
        st.subheader("Chat with your Data")
        
        # Display chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat Input
        if user_query := st.chat_input("Ex: 'Which region had the highest sales growth?'"):
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            # Execution of RAG flow
            context = get_data_context(df)
            response = bi_chain.run(context=context, question=user_query)

            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

    with tab2:
        st.subheader("Automated Data Visualizations")
        
        col1, col2 = st.columns(2)
        
        # Determine numeric and categorical columns for plotting
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

        if len(numeric_cols) >= 1 and len(categorical_cols) >= 1:
            with col1:
                fig1 = px.bar(df, x=categorical_cols[0], y=numeric_cols[0], 
                             title=f"{numeric_cols[0]} by {categorical_cols[0]}",
                             color_discrete_sequence=['#636EFA'])
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Assuming there's a 'Date' or time-based column for trends
                date_col = next((c for c in df.columns if 'date' in c.lower()), None)
                if date_col:
                    df[date_col] = pd.to_datetime(df[date_col])
                    fig2 = px.line(df.sort_values(date_col), x=date_col, y=numeric_cols[0], 
                                  title="Performance Trend Over Time")
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.info("Upload a dataset with a 'Date' column to see time-series trends.")
        
        st.dataframe(df.head(10))

else:
    st.info("Please upload a CSV file in the sidebar to begin analysis.")