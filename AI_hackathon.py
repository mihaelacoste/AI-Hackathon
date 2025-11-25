# AI_hackathon.py - Core Expense Tracker Logic and Gemini Agent Integration (Final Plotly Version)

import datetime
import re
import csv
import os
import pandas as pd
import matplotlib.pyplot as plt 
import requests
import json
import time
import plotly.express as px # NEW: For interactive charts

# Set Matplotlib to a non-interactive backend for environment flexibility
plt.switch_backend('Agg') 
plt.style.use('ggplot')

# --- 1. GLOBAL DATA & UTILITIES ---

expenses = []
expense_id_counter = 1

def get_next_id():
    """Returns the next available unique ID."""
    global expense_id_counter
    next_id = expense_id_counter
    expense_id_counter += 1
    return next_id

def format_date(date_str):
    """Utility to ensure date is in 'YYYY-MM-DD' format."""
    try:
        return datetime.datetime.strptime(date_str, '%Y-%m-%d').strftime('%Y-%m-%d')
    except ValueError:
        return datetime.date.today().strftime('%Y-%m-%d')

# --- 2. CORE CRUD FUNCTIONS ---
def add_expense(amount: float, category: str, description: str, date: str, tags: list = None):
    """Adds a new expense to the in-memory list."""
    new_expense = {
        'id': get_next_id(),
        'amount': round(float(amount), 2),
        'category': category.strip().lower(),
        'date': format_date(date),
        'description': description.strip(),
        'tags': [t.strip().lower() for t in (tags if tags else [])]
    }
    expenses.append(new_expense)
    print(f"Expense #{new_expense['id']} ({new_expense['description']}) added successfully.")
    return new_expense

def list_expenses(expense_list=None):
    """Displays expenses in a formatted table (prints to console/logs)."""
    data_to_show = expense_list if expense_list is not None else expenses
    
    if not data_to_show:
        print("🔍 No expenses to display.")
        return

    header = f"{'ID':<4} | {'Date':<10} | {'Amount':>8} | {'Category':<15} | {'Description':<30} | Tags"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    
    for exp in data_to_show:
        tags_str = ", ".join(exp['tags'])
        print(f"{exp['id']:<4} | {exp['date']:<10} | ${exp['amount']:>7.2f} | {exp['category']:<15} | {exp['description']:<30} | {tags_str}")
    print("=" * len(header))
    
def filter_expenses(category: str = None, tag: str = None, start_date: str = None, end_date: str = None):
    """Filters the expenses list based on provided criteria."""
    filtered_list = expenses

    if category:
        cat_lower = category.strip().lower()
        filtered_list = [exp for exp in filtered_list if exp['category'] == cat_lower]

    if tag:
        tag_lower = tag.strip().lower()
        filtered_list = [exp for exp in filtered_list if tag_lower in exp['tags']]

    if start_date:
        start_date = format_date(start_date)
        filtered_list = [exp for exp in filtered_list if exp['date'] >= start_date]

    if end_date:
        end_date = format_date(end_date)
        filtered_list = [exp for exp in filtered_list if exp['date'] <= end_date]

    print(f"\n--- Filter Results ({len(filtered_list)} items found) ---")
    list_expenses(filtered_list)
    
    total = sum(exp['amount'] for exp in filtered_list)
    return f"Filter complete. Found {len(filtered_list)} expenses totaling ${total:.2f}."

def create_dataframe(expense_list=None):
    """Converts the list of dictionaries into a Pandas DataFrame."""
    data = expense_list if expense_list is not None else expenses
    if not data:
        print("⚠️ No data available to create a DataFrame.")
        return None
    
    df = pd.DataFrame(data)
    
    if 'amount' in df.columns and not df['amount'].empty:
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        df.dropna(subset=['amount'], inplace=True)
    
    if 'date' in df.columns and not df['date'].empty:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df.dropna(subset=['date'], inplace=True)
    
    return df

def plot_category_distribution(**kwargs):
    """Generates and returns a Plotly Figure object for distribution."""
    df = create_dataframe()
    if df is None or df.empty:
        return "Not enough data to generate category distribution chart."

    # Group data for Plotly
    category_spending = df.groupby('category')['amount'].sum().reset_index()
    category_spending.columns = ['Category', 'Amount']
    
    # Generate Plotly Pie Chart
    fig = px.pie(
        category_spending,
        values='Amount',
        names='Category',
        title='Expense Distribution by Category',
        hole=.3,
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    
    # Improve layout for presentation
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')

    plt.close('all') # Aggressive cleanup for Streamlit stability
    return fig

def plot_spending_trend(**kwargs):
    """Generates and returns a Plotly Figure object for spending trend."""
    df = create_dataframe()
    if df is None or df.empty:
        return "Not enough data to generate spending trend chart."

    daily_spending = df.groupby('date')['amount'].sum().reset_index()
    daily_spending.columns = ['Date', 'Amount']

    # Generate Plotly Line Chart
    fig = px.line(
        daily_spending,
        x='Date',
        y='Amount',
        title='Daily Spending Trend Over Time',
        markers=True
    )
    
    fig.update_traces(line_color='#0066CC')
    
    plt.close('all') # Aggressive cleanup for Streamlit stability
    return fig

# --- 4. GEMINI AGENT LOGIC (Zero-Shot Approach) ---

TOOL_MAP = {
    "filter": filter_expenses,
    "plot_distribution": plot_category_distribution,
    "plot_trend": plot_spending_trend,
}

INTENT_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "intent": {"type": "STRING", "description": "One of: 'plot_distribution', 'plot_trend', 'filter', or 'none'."},
        "category": {"type": "STRING", "description": "The category to filter by (e.g., 'groceries', 'transport')."},
        "start_date": {"type": "STRING", "description": "Start date for filter in YYYY-MM-DD format."},
        "end_date": {"type": "STRING", "description": "End date for filter in YYYY-MM-DD format."}
    },
    "required": ["intent"]
}


def parse_expenses_via_gemini(user_prompt: str, api_key: str):
    """Uses Gemini's structured output to extract multiple expenses."""
    if not api_key:
        return None, "API Key is missing."

    system_prompt = (
        "You are an expense parser. Your task is to extract all individual expenses from the user's "
        "raw text input, even if the user provides a list (e.g., 'potatoes 2.50\nlimes 3.32'). "
        "For each item, determine the amount, a suitable financial category (e.g., Groceries, Transport, Bills), "
        "a description, and the date (default to today if not provided). "
        "ALWAYS respond with a JSON array that strictly matches the required schema. If no expenses are found, return an empty array."
    )

    response_schema = {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "amount": {"type": "number", "description": "The monetary amount."},
                "category": {"type": "string", "description": "The financial category."},
                "description": {"type": "string", "description": "A short, descriptive name."},
                "date": {"type": "string", "description": "Date in YYYY-MM-DD format."}
            },
            "required": ["amount", "category", "description", "date"]
        }
    }

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={api_key}"

    payload = {
        "contents": [{"parts": [{"text": user_prompt}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": response_schema
        }
    }

    try:
        response = requests.post(url, headers={'Content-Type': 'application/json'}, json=payload)
        response.raise_for_status()
        
        result = response.json()
        json_text = result['candidates'][0]['content']['parts'][0]['text'].strip()
        expenses_list = json.loads(json_text)
        
        if not isinstance(expenses_list, list):
            return None, f"⚠️ Gemini returned non-list JSON: {json_text}"

        parsed_count = 0
        for item in expenses_list:
            add_expense(
                amount=item.get('amount'),
                category=item.get('category'),
                description=item.get('description'),
                date=item.get('date', datetime.date.today().strftime('%Y-%m-%d'))
            )
            parsed_count += 1
            
        return expenses_list, f"Successfully parsed and added {parsed_count} expenses."

    except Exception as e:
        print(f"Error parsing expenses via Gemini: {e}")
        return None, f"An error occurred during parsing: {e}"


def execute_agent_command(user_prompt: str, api_key: str) -> tuple:
    """
    Uses Zero-Shot Text Generation to identify intent and execute the corresponding function locally.
    Returns (markdown_response, image_figure_or_none).
    """
    if not api_key:
        return "API Key is missing.", None

    system_prompt = (
        "You are an intelligent data agent. Analyze the user's request and determine the user's intent: "
        "'plot_distribution' (for pie/bar charts), 'plot_trend' (for line charts), 'filter' (for specific data requests), or 'none' (for greetings/other questions). "
        "Extract any relevant parameters like category or dates. Respond ONLY with a single JSON object matching the required schema. Do not output any other text."
    )
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={api_key}"
    
    payload = {
        "contents": [{"parts": [{"text": user_prompt}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": INTENT_SCHEMA
        }
    }

    try:
        response = requests.post(url, headers={'Content-Type': 'application/json'}, json=payload)
        response.raise_for_status()
        
        result = response.json()
        json_text = result['candidates'][0]['content']['parts'][0]['text'].strip()
        intent_data = json.loads(json_text)
        
        intent = intent_data.get('intent', 'none').lower()
        args = {k: v for k, v in intent_data.items() if k != 'intent' and v is not None}
        
        if intent in TOOL_MAP:
            if not expenses:
                return f"Tool '{intent}' could not run. No expenses have been added yet.", None
                
            tool_output = TOOL_MAP[intent](**args)
            
            if intent.startswith('plot'):
                # Chart functions return the Plotly figure object
                if isinstance(tool_output, str): # Error message from plotting function
                    return f"❌ Charting Failed: {tool_output}", None
                return f"✅ Done! Chart generated successfully.", tool_output
            else:
                # Filter functions return a summary string
                return f"✅ Done! {tool_output}", None
        
        elif intent == 'none' or intent == 'unhandled':
            natural_payload = {
                "contents": [{"parts": [{"text": "Generate a short, friendly response to this query: " + user_prompt}]}]
            }
            response = requests.post(url, headers={'Content-Type': 'application/json'}, json=natural_payload)
            response.raise_for_status()
            return response.json()['candidates'][0]['content']['parts'][0]['text'], None
        
        return f"Agent could not map '{intent}' to an action.", None

    except Exception as e:
        print(f"FATAL AGENT COMMAND ERROR: {e}")
        return f"An internal error occurred: {e}", None


# --- SAFE TEST EXECUTION BLOCK ---
if __name__ == "__main__":
    print("\n--- Running standalone tests in AI_hackathon.py ---")