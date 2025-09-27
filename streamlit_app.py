"""
Multi-Agent AI System - Streamlit Web Interface
Company Sales Data Analysis Dashboard
"""

import streamlit as st
import pandas as pd
from groq import Groq
from datetime import datetime
import json
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Multi-Agent Sales Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class MultiAgentOrchestrator:
    def __init__(self, api_key):
        """Initialize the orchestrator with Groq API"""
        self.client = Groq(api_key=api_key)
        self.data = None
        self.company_email = "rehanhuzaifa96@gmail.com"
        self.boss_email = "huzaifagg36@gmail.com"
        
    def load_data(self):
        """Load the company sales data"""
        try:
            self.data = pd.read_csv('company_sales.csv')
            self.data['Date'] = pd.to_datetime(self.data['Date'])
            self.data['Profit'] = self.data['Revenue'] - self.data['Expenses']
            self.data['Month'] = self.data['Date'].dt.strftime('%B')
            return True
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return False
    
    def get_analysis_data(self):
        """Get basic analysis without AI"""
        if self.data is None:
            return None
            
        total_revenue = self.data['Revenue'].sum()
        total_expenses = self.data['Expenses'].sum()
        total_profit = self.data['Profit'].sum()
        avg_customers = self.data['Customers'].mean()
        
        # Product performance
        product_performance = self.data.groupby('Product').agg({
            'Revenue': 'sum',
            'Expenses': 'sum',
            'Profit': 'sum',
            'Customers': 'sum'
        }).round(2)
        
        # Monthly trends
        monthly_trends = self.data.groupby('Month').agg({
            'Revenue': 'sum',
            'Profit': 'sum',
            'Customers': 'sum'
        }).round(2)
        
        best_month = monthly_trends['Revenue'].idxmax()
        best_product = product_performance['Revenue'].idxmax()
        
        return {
            'total_revenue': total_revenue,
            'total_expenses': total_expenses,
            'total_profit': total_profit,
            'profit_margin': round((total_profit/total_revenue)*100, 2),
            'avg_customers': round(avg_customers, 0),
            'best_month': best_month,
            'best_product': best_product,
            'product_performance': product_performance,
            'monthly_trends': monthly_trends
        }
    
    def data_analyst_agent(self, analysis_data):
        """Agent 1: AI-powered data analysis"""
        prompt = f"""
        As a Data Analyst Agent, analyze this company sales data and provide insights:
        
        Key Metrics:
        - Total Revenue: ${analysis_data['total_revenue']:,}
        - Total Expenses: ${analysis_data['total_expenses']:,}
        - Total Profit: ${analysis_data['total_profit']:,}
        - Profit Margin: {analysis_data['profit_margin']}%
        - Average Customers per Month: {analysis_data['avg_customers']:.0f}
        
        Product Performance:
        {analysis_data['product_performance'].to_string()}
        
        Best Performing Month: {analysis_data['best_month']}
        Best Performing Product: {analysis_data['best_product']}
        
        Provide detailed insights about:
        1. Revenue and profit trends
        2. Product performance comparison
        3. Customer acquisition patterns
        4. Seasonal trends
        5. Areas of concern or opportunity
        
        Include specific numbers and calculations in your analysis.
        """
        
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
                temperature=0.7
            )
            analysis_data['ai_insights'] = response.choices[0].message.content
            return analysis_data
        except Exception as e:
            analysis_data['ai_insights'] = f"Error generating AI insights: {e}"
            return analysis_data
    
    def summarizer_agent(self, analyst_output, analysis_data):
        """Agent 2: Summarize insights into bullet points"""
        prompt = f"""
        As a Summarizer Agent, take this detailed analysis and create 4-5 clear, concise bullet points:
        
        Analysis Data:
        - Total Revenue: ${analysis_data['total_revenue']:,}
        - Total Profit: ${analysis_data['total_profit']:,}
        - Profit Margin: {analysis_data['profit_margin']}%
        - Best Month: {analysis_data['best_month']}
        - Best Product: {analysis_data['best_product']}
        
        Detailed Insights:
        {analyst_output}
        
        Create exactly 4-5 bullet points that capture the most important insights.
        Make them simple, clear, and actionable for business decision-making.
        """
        
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating summary: {e}"
    
    def business_strategy_agent(self, summary_output):
        """Agent 3: Generate business strategies"""
        prompt = f"""
        As a Business Strategy Agent, based on these key insights, recommend practical strategies:
        
        Key Insights:
        {summary_output}
        
        Provide specific, actionable recommendations for:
        1. Increasing sales revenue
        2. Reducing operational costs
        3. Growing customer base
        4. Improving profit margins
        5. Seasonal optimization
        
        Make each recommendation practical and implementable with clear next steps.
        """
        
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating strategy: {e}"
    
    def customer_support_agent(self, question):
        """Agent 4: Handle customer queries"""
        data_summary = f"""
        Dataset Overview:
        - Time Period: January 2024 - December 2024
        - Products: {', '.join(self.data['Product'].unique())}
        - Total Records: {len(self.data)}
        
        Key Statistics:
        - Total Revenue: ${self.data['Revenue'].sum():,}
        - Total Profit: ${self.data['Profit'].sum():,}
        - Average Monthly Customers: {self.data['Customers'].mean():.0f}
        
        Full Dataset:
        {self.data.to_string()}
        """
        
        prompt = f"""
        As a friendly Customer Support Agent, answer this question about our company sales data:
        
        Question: {question}
        
        Data Context:
        {data_summary}
        
        Provide a friendly, accurate, and helpful response. Include specific numbers when relevant.
        """
        
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error processing question: {e}"
    
    def email_reporter_agent(self, summary_output, strategy_output):
        """Agent 5: Generate email report"""
        current_date = datetime.now().strftime("%B %d, %Y")
        
        prompt = f"""
        Create a professional email for the company boss summarizing monthly performance:
        
        Key Insights Summary:
        {summary_output}
        
        Strategic Recommendations:
        {strategy_output}
        
        Create a formal business email with:
        - Professional greeting to the boss
        - Executive summary of performance
        - Key insights (3-4 main points)
        - Strategic recommendations (2-3 top priorities)
        - Professional closing
        
        Date: {current_date}
        Make it concise but comprehensive, suitable for a busy executive.
        """
        
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating email: {e}"
    
    def prepare_email_content(self, email_content):
        """Prepare email content for copying"""
        subject = "Monthly Sales Performance Report - " + datetime.now().strftime("%B %Y")
        
        formatted_email = f"""Subject: {subject}
From: {self.company_email}
To: {self.boss_email}

{email_content}

---
Generated by Multi-Agent AI Sales Analytics System
Date: {datetime.now().strftime("%B %d, %Y at %I:%M %p")}
"""
        return formatted_email

def create_charts(data):
    """Create interactive charts for the dashboard"""
    
    # Revenue by Product
    product_revenue = data.groupby('Product')['Revenue'].sum().reset_index()
    fig1 = px.bar(product_revenue, x='Product', y='Revenue', 
                  title='Total Revenue by Product',
                  color='Revenue', color_continuous_scale='Blues')
    
    # Monthly Revenue Trend
    monthly_data = data.groupby('Month').agg({
        'Revenue': 'sum',
        'Profit': 'sum',
        'Customers': 'sum'
    }).reset_index()
    
    fig2 = px.line(monthly_data, x='Month', y='Revenue', 
                   title='Monthly Revenue Trend',
                   markers=True)
    
    # Profit vs Revenue
    fig3 = px.scatter(data, x='Revenue', y='Profit', 
                      color='Product', size='Customers',
                      title='Profit vs Revenue by Product')
    
    return fig1, fig2, fig3

def main():
    """Main Streamlit application"""
    
    # Header
    st.title("🤖 Multi-Agent Sales Analytics Dashboard")
    st.markdown("**AI-Powered Company Sales Data Analysis System**")
    
    # Initialize session state
    if 'orchestrator' not in st.session_state:
        API_KEY = "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"
        st.session_state.orchestrator = MultiAgentOrchestrator(API_KEY)
        
    # Load data
    if st.session_state.orchestrator.load_data():
        data = st.session_state.orchestrator.data
        
        # Sidebar
        st.sidebar.title("🎛️ Control Panel")
        
        # Navigation
        page = st.sidebar.selectbox(
            "Choose Analysis Type:",
            ["📊 Dashboard Overview", "🤖 Full AI Analysis", "🎧 Customer Support", "📧 Email Report"]
        )
        
        if page == "📊 Dashboard Overview":
            st.header("📊 Sales Data Overview")
            
            # Get basic analysis
            analysis_data = st.session_state.orchestrator.get_analysis_data()
            
            # Key Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Revenue", f"${analysis_data['total_revenue']:,}")
            with col2:
                st.metric("Total Profit", f"${analysis_data['total_profit']:,}")
            with col3:
                st.metric("Profit Margin", f"{analysis_data['profit_margin']}%")
            with col4:
                st.metric("Avg Customers", f"{analysis_data['avg_customers']:.0f}")
            
            # Charts
            st.subheader("📈 Visual Analytics")
            fig1, fig2, fig3 = create_charts(data)
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig1, use_container_width=True)
            with col2:
                st.plotly_chart(fig2, use_container_width=True)
            
            st.plotly_chart(fig3, use_container_width=True)
            
            # Data Tables
            st.subheader("📋 Performance Tables")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Product Performance**")
                st.dataframe(analysis_data['product_performance'])
            with col2:
                st.write("**Monthly Trends**")
                st.dataframe(analysis_data['monthly_trends'])
        
        elif page == "🤖 Full AI Analysis":
            st.header("🤖 AI-Powered Multi-Agent Analysis")
            
            if st.button("🚀 Run Complete Analysis", type="primary"):
                with st.spinner("Running multi-agent analysis..."):
                    
                    # Get analysis data
                    analysis_data = st.session_state.orchestrator.get_analysis_data()
                    
                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Agent 1: Data Analyst
                    status_text.text("📊 Data Analyst Agent working...")
                    progress_bar.progress(20)
                    analyst_output = st.session_state.orchestrator.data_analyst_agent(analysis_data)
                    
                    # Agent 2: Summarizer
                    status_text.text("📝 Summarizer Agent working...")
                    progress_bar.progress(40)
                    summary_output = st.session_state.orchestrator.summarizer_agent(analyst_output, analysis_data)
                    
                    # Agent 3: Strategy
                    status_text.text("💡 Business Strategy Agent working...")
                    progress_bar.progress(60)
                    strategy_output = st.session_state.orchestrator.business_strategy_agent(summary_output)
                    
                    # Agent 4: Email
                    status_text.text("📧 Email Reporter Agent working...")
                    progress_bar.progress(80)
                    email_output = st.session_state.orchestrator.email_reporter_agent(summary_output, strategy_output)
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Analysis Complete!")
                    
                    # Display results
                    st.success("🎉 Multi-Agent Analysis Complete!")
                    
                    # Tabs for organized display
                    tab1, tab2, tab3, tab4 = st.tabs(["📊 Analysis", "📝 Summary", "💡 Strategy", "📧 Email"])
                    
                    with tab1:
                        st.subheader("📊 Data Analyst Insights")
                        st.write(analyst_output)
                        
                        # Key metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Revenue", f"${analysis_data['total_revenue']:,}")
                        with col2:
                            st.metric("Profit", f"${analysis_data['total_profit']:,}")
                        with col3:
                            st.metric("Best Product", analysis_data['best_product'])
                    
                    with tab2:
                        st.subheader("📝 Executive Summary")
                        st.write(summary_output)
                    
                    with tab3:
                        st.subheader("💡 Business Strategy Recommendations")
                        st.write(strategy_output)
                    
                    with tab4:
                        st.subheader("📧 Executive Email Report")
                        st.write(email_output)
                        
                        # Email sending option
                        if st.button("📤 Send Email to Boss"):
                            st.info("Email functionality ready! (Configure SMTP settings to enable sending)")
                            st.code(f"To: huzaifagg36@gmail.com\nFrom: rehanhuzaifa96@gmail.com\n\n{email_output}")
        
        elif page == "🎧 Customer Support":
            st.header("🎧 Customer Support Agent")
            st.write("Ask any question about the sales data!")
            
            # Sample questions
            st.subheader("💡 Sample Questions:")
            sample_questions = [
                "Which product had the highest revenue in October?",
                "What was our total profit in Q4?",
                "How many customers did we have for Tablets?",
                "What's our best performing month?",
                "Compare revenue between Laptop and Phone products"
            ]
            
            for i, question in enumerate(sample_questions):
                if st.button(f"❓ {question}", key=f"sample_{i}"):
                    with st.spinner("Customer Support Agent thinking..."):
                        answer = st.session_state.orchestrator.customer_support_agent(question)
                        st.success("🎧 Customer Support Response:")
                        st.write(answer)
            
            # Custom question
            st.subheader("🗣️ Ask Your Own Question:")
            user_question = st.text_input("Enter your question about the sales data:")
            
            if st.button("🔍 Get Answer") and user_question:
                with st.spinner("Processing your question..."):
                    answer = st.session_state.orchestrator.customer_support_agent(user_question)
                    st.success("🎧 Customer Support Response:")
                    st.write(answer)
        
        elif page == "📧 Email Report":
            st.header("📧 Email Report Generator")
            
            if st.button("📝 Generate Email Report", type="primary"):
                with st.spinner("Generating email report..."):
                    # Get data and run agents
                    analysis_data = st.session_state.orchestrator.get_analysis_data()
                    analyst_output = st.session_state.orchestrator.data_analyst_agent(analysis_data)
                    summary_output = st.session_state.orchestrator.summarizer_agent(analyst_output, analysis_data)
                    strategy_output = st.session_state.orchestrator.business_strategy_agent(summary_output)
                    email_output = st.session_state.orchestrator.email_reporter_agent(summary_output, strategy_output)
                    
                    st.success("📧 Email Report Generated!")
                    
                    # Email details
                    st.subheader("📬 Email Details")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**From:** rehanhuzaifa96@gmail.com")
                        st.write("**To:** huzaifagg36@gmail.com")
                    with col2:
                        st.write("**Subject:** Monthly Sales Performance Report")
                        st.write("**Date:** " + datetime.now().strftime("%B %d, %Y"))
                    
                    # Email content
                    st.subheader("📄 Email Content")
                    st.text_area("Email Body", email_output, height=400)
                    
                    # Email sending section
                    st.subheader("📤 Send Email")
                    
                    # Email credentials input
                    with st.expander("📧 Email Settings (Required to send email)"):
                        sender_email = st.text_input("Your Gmail Address:", value="rehanhuzaifa96@gmail.com")
                        sender_password = st.text_input("Gmail App Password:", type="password", 
                                                      help="Use Gmail App Password, not your regular password")
                        recipient_email = st.text_input("Boss Email:", value="huzaifagg36@gmail.com")
                        
                        st.info("💡 **How to get Gmail App Password:**\n"
                               "1. Go to Google Account settings\n"
                               "2. Security → 2-Step Verification → App passwords\n"
                               "3. Generate app password for 'Mail'\n"
                               "4. Use that 16-character password here")
                    
                    # Prepare email for copying
                    formatted_email = st.session_state.orchestrator.prepare_email_content(email_output)
                    
                    st.subheader("📋 Copy Email Content")
                    st.text_area("Ready to Copy & Send:", formatted_email, height=300)
                    
                    st.info("📧 **How to Send:**\n"
                           "1. Copy the email content above\n"
                           "2. Open your Gmail (rehanhuzaifa96@gmail.com)\n"
                           "3. Compose new email to huzaifagg36@gmail.com\n"
                           "4. Paste the content and send!")
                    
                    if st.button("📤 Copy to Clipboard", type="primary"):
                        st.success("✅ Email content ready to copy!")
                        st.balloons()
        
        # Raw data view
        with st.sidebar:
            st.subheader("📋 Raw Data")
            if st.checkbox("Show Raw Data"):
                st.dataframe(data)
    
    else:
        st.error("❌ Could not load sales data. Please check company_sales.csv file.")

if __name__ == "__main__":
    main()