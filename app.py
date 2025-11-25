# app.py - AI Expense Tracker Streamlit UI (Final Plotly Version)

import streamlit as st
import pandas as pd
import os 
import time
import matplotlib.pyplot as plt # Still needed for plt.close('all')

# --- IMPORT ALL NECESSARY LOGIC FROM AI_hackathon.py ---
try:
    import AI_hackathon as AI_LOGIC
except ImportError:
    st.error("Error: Could not find 'AI_hackathon.py'. Please ensure your backend file is named correctly and is in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"Error during import: {e}")
    st.stop()


# --- 0. SESSION STATE INITIALIZATION ---
if 'expense_input_content' not in st.session_state:
    st.session_state.expense_input_content = ""
if 'agent_query_content' not in st.session_state:
    st.session_state.agent_query_content = ""
if 'gemini_api_key' not in st.session_state:
    st.session_state.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
if 'chart_fig' not in st.session_state:
    st.session_state.chart_fig = None


# --- UTILITY FUNCTION FOR STREAMLIT DISPLAY ---
@st.cache_data(show_spinner=False)
def display_expenses_dataframe():
    """Converts the global expense list to a DataFrame and displays it."""
    if 'AI_LOGIC' in globals():
        df = AI_LOGIC.create_dataframe()
        if df is not None and not df.empty:
            return df.set_index('id')
    return None


# --- STREAMLIT APP CONFIGURATION ---
st.set_page_config(layout="wide", page_title="AI Expense Tracker")
st.title("AI Expense Tracker Hackathon Project")

# --- 1. API Key Status (Secure Display) ---
# st.subheader("API Status")
if st.session_state.gemini_api_key:
    # st.success("✅ Gemini API Key loaded securely from GEMINI_API_KEY environment variable.")
    is_api_key_set = True
else:
    # st.error("❌ GEMINI_API_KEY environment variable not set. Please set it in your terminal to enable AI features.")
    is_api_key_set = False


# --- 2. Add Expenses or Run Command (Submission Logic) ---
st.header("Add Expenses (List or natural language input)")

expense_input_content = st.text_area(
    "Enter expense list (one per line, e.g., 'potatoes 2.50') or natural language command like 'I spent $15 on groceries on 24 October 2025'", 
    height=100,
    value=st.session_state.expense_input_content,
    key="expense_input_key"
)

if st.button("Submit", use_container_width=True, disabled=not is_api_key_set):
    current_input = st.session_state.expense_input_key.strip()
    
    if is_api_key_set and current_input:
        display_expenses_dataframe.clear() # Clear cache BEFORE processing
        st.session_state.chart_fig = None # Clear any previous chart
        with st.spinner('Agent analyzing input via Gemini...'):
            try:
                # Use the structured parser to extract expenses
                results, message = AI_LOGIC.parse_expenses_via_gemini(
                    current_input, 
                    api_key=st.session_state.gemini_api_key
                )
                
                if results is not None:
                    st.success(message)
                else:
                    st.warning(message)
                    
            except Exception as e:
                st.error(f"❌ API Error during parsing: {e}")
                
        # Clear the input area and refresh the DataFrame cache
        st.session_state.expense_input_content = ""
        st.rerun()
    elif not current_input:
        st.warning("Please enter some text.")


# --- 3. Current Expenses View ---
st.header(" Current Expenses")
df_display = display_expenses_dataframe()
if df_display is not None:
    st.dataframe(df_display)
else:
    st.info("No expenses recorded yet.")


# --- 4. Agent Command Interface (Zero-Shot Text Mode) ---
st.header("Run Visualization & Filtering Commands")

# Initialize agent output for this section
if 'zero_shot_agent_output' not in st.session_state:
    st.session_state.zero_shot_agent_output = ""

st.info(f"🤖 Agent Response: {st.session_state.zero_shot_agent_output}")

agent_query_content = st.text_input(
    "Ask the agent a question or request a chart:", 
    placeholder="Show me the grocery expenses or Generate a pie chart of spending.",
    value=st.session_state.agent_query_content,
    key="agent_query_key"
)

if st.button("Ask for a filter or line/pie chart", use_container_width=True, disabled=not is_api_key_set):
    current_query = st.session_state.agent_query_key.strip()
    
    # Reset chart state at start of command
    st.session_state.chart_fig = None 
    
    if is_api_key_set and current_query:
        display_expenses_dataframe.clear() # Clear data cache for fresh view
        with st.spinner('Agent thinking and executing command...'):
            try:
                # CALL NEW ZERO-SHOT COMMAND FUNCTION
                message, image_figure = AI_LOGIC.execute_agent_command(
                    current_query, 
                    api_key=st.session_state.gemini_api_key
                )
                
                st.session_state.zero_shot_agent_output = message
                
                # If a figure object was returned, store it in state
                if image_figure is not None and not isinstance(image_figure, str):
                    st.session_state.chart_fig = image_figure
                    
            except Exception as e:
                st.error(f"❌ API Error during command execution: {e}")
                
        # Clear the query input and refresh data
        st.session_state.agent_query_content = ""
        st.rerun()
    elif not current_query:
        st.warning("Please enter a query.")

# --- CHART DISPLAY (Always try to display state if available) ---
if st.session_state.chart_fig is not None:
    st.subheader("Generated Chart")
    try:
        # Use st.plotly_chart for the interactive Plotly figure
        st.plotly_chart(st.session_state.chart_fig, use_container_width=True)
        plt.close('all') # Essential cleanup for Matplotlib/Streamlit stability
    except Exception as e:
        st.error(f"Error displaying chart: {e}") 