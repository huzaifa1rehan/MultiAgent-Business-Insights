"""
Multi-Agent AI System Orchestrator
Manages specialized agents for company sales data analysis
"""

import pandas as pd
from groq import Groq
from datetime import datetime
import json

class MultiAgentOrchestrator:
    def __init__(self, api_key):
        """Initialize the orchestrator with Groq API"""
        self.client = Groq(api_key=api_key)
        self.data = None
        self.load_data()
        
    def load_data(self):
        """Load the company sales data"""
        try:
            self.data = pd.read_csv('company_sales.csv')
            self.data['Date'] = pd.to_datetime(self.data['Date'])
            self.data['Profit'] = self.data['Revenue'] - self.data['Expenses']
            print("✅ Sales data loaded successfully")
        except Exception as e:
            print(f"❌ Error loading data: {e}")
    
    def run_full_analysis(self):
        """Execute the complete workflow: Analyst → Summarizer → Strategy → Email"""
        print("\n🚀 Starting Multi-Agent Analysis Workflow...")
        
        # Step 1: Data Analyst Agent
        print("\n📊 Running Data Analyst Agent...")
        analyst_output = self.data_analyst_agent()
        
        # Step 2: Summarizer Agent
        print("\n📝 Running Summarizer Agent...")
        summary_output = self.summarizer_agent(analyst_output)
        
        # Step 3: Business Strategy Agent
        print("\n💡 Running Business Strategy Agent...")
        strategy_output = self.business_strategy_agent(summary_output)
        
        # Step 4: Email Reporter Agent
        print("\n📧 Running Email Reporter Agent...")
        email_output = self.email_reporter_agent(summary_output, strategy_output)
        
        return {
            'analysis': analyst_output,
            'summary': summary_output,
            'strategy': strategy_output,
            'email': email_output
        }
    
    def customer_support_query(self, question):
        """Handle customer support queries independently"""
        print(f"\n🎧 Customer Support Agent handling: {question}")
        return self.customer_support_agent(question)
    
    def data_analyst_agent(self):
        """Agent 1: Analyze sales data and generate insights"""
        # Perform calculations
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
        self.data['Month'] = self.data['Date'].dt.strftime('%B')
        monthly_trends = self.data.groupby('Month').agg({
            'Revenue': 'sum',
            'Profit': 'sum'
        }).round(2)
        
        # Best performing month and product
        best_month = monthly_trends['Revenue'].idxmax()
        best_product = product_performance['Revenue'].idxmax()
        
        analysis_data = {
            'total_revenue': total_revenue,
            'total_expenses': total_expenses,
            'total_profit': total_profit,
            'profit_margin': round((total_profit/total_revenue)*100, 2),
            'avg_customers': round(avg_customers, 0),
            'best_month': best_month,
            'best_product': best_product,
            'product_performance': product_performance.to_dict(),
            'monthly_trends': monthly_trends.to_dict()
        }
        
        # Generate AI insights
        prompt = f"""
        As a Data Analyst Agent, analyze this company sales data and provide insights:
        
        Key Metrics:
        - Total Revenue: ${total_revenue:,}
        - Total Expenses: ${total_expenses:,}
        - Total Profit: ${total_profit:,}
        - Profit Margin: {analysis_data['profit_margin']}%
        - Average Customers per Month: {avg_customers:.0f}
        
        Product Performance:
        {product_performance.to_string()}
        
        Best Performing Month: {best_month}
        Best Performing Product: {best_product}
        
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
                model="llama3-8b-8192",
                temperature=0.7
            )
            analysis_data['ai_insights'] = response.choices[0].message.content
            return analysis_data
        except Exception as e:
            analysis_data['ai_insights'] = f"Error generating AI insights: {e}"
            return analysis_data
    
    def summarizer_agent(self, analyst_output):
        """Agent 2: Summarize insights into 4-5 bullet points"""
        prompt = f"""
        As a Summarizer Agent, take this detailed analysis and create 4-5 clear, concise bullet points:
        
        Analysis Data:
        - Total Revenue: ${analyst_output['total_revenue']:,}
        - Total Profit: ${analyst_output['total_profit']:,}
        - Profit Margin: {analyst_output['profit_margin']}%
        - Best Month: {analyst_output['best_month']}
        - Best Product: {analyst_output['best_product']}
        
        Detailed Insights:
        {analyst_output['ai_insights']}
        
        Create exactly 4-5 bullet points that capture the most important insights.
        Make them simple, clear, and actionable for business decision-making.
        """
        
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-8b-8192",
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating summary: {e}"
    
    def business_strategy_agent(self, summary_output):
        """Agent 3: Generate practical business strategies"""
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
                model="llama3-8b-8192",
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating strategy: {e}"
    
    def customer_support_agent(self, question):
        """Agent 4: Handle customer queries about the dataset"""
        # Prepare data context for the AI
        data_summary = f"""
        Dataset Overview:
        - Time Period: January 2024 - December 2024
        - Products: {', '.join(self.data['Product'].unique())}
        - Total Records: {len(self.data)}
        
        Sample Data:
        {self.data.head().to_string()}
        
        Key Statistics:
        - Total Revenue: ${self.data['Revenue'].sum():,}
        - Total Profit: ${self.data['Profit'].sum():,}
        - Average Monthly Customers: {self.data['Customers'].mean():.0f}
        """
        
        prompt = f"""
        As a friendly Customer Support Agent, answer this question about our company sales data:
        
        Question: {question}
        
        Data Context:
        {data_summary}
        
        Full Dataset:
        {self.data.to_string()}
        
        Provide a friendly, accurate, and helpful response. Include specific numbers when relevant.
        If the question can't be answered with the available data, politely explain what information is available.
        """
        
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-8b-8192",
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"I apologize, but I encountered an error processing your question: {e}"
    
    def email_reporter_agent(self, summary_output, strategy_output):
        """Agent 5: Generate professional email report"""
        current_date = datetime.now().strftime("%B %d, %Y")
        
        prompt = f"""
        As an Email Reporter Agent, create a professional email for the company boss summarizing monthly performance:
        
        Key Insights Summary:
        {summary_output}
        
        Strategic Recommendations:
        {strategy_output}
        
        Create a formal business email with:
        - Professional greeting
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
                model="llama3-8b-8192",
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating email report: {e}"
    
    def display_results(self, results):
        """Display all results in a formatted way"""
        print("\n" + "="*80)
        print("📊 MULTI-AGENT ANALYSIS RESULTS")
        print("="*80)
        
        print("\n📈 DATA ANALYSIS INSIGHTS:")
        print("-" * 50)
        if 'ai_insights' in results['analysis']:
            print(results['analysis']['ai_insights'])
        
        print(f"\n💰 Key Metrics:")
        print(f"• Total Revenue: ${results['analysis']['total_revenue']:,}")
        print(f"• Total Profit: ${results['analysis']['total_profit']:,}")
        print(f"• Profit Margin: {results['analysis']['profit_margin']}%")
        print(f"• Best Month: {results['analysis']['best_month']}")
        print(f"• Best Product: {results['analysis']['best_product']}")
        
        print("\n📝 EXECUTIVE SUMMARY:")
        print("-" * 50)
        print(results['summary'])
        
        print("\n💡 BUSINESS STRATEGY RECOMMENDATIONS:")
        print("-" * 50)
        print(results['strategy'])
        
        print("\n📧 EMAIL REPORT:")
        print("-" * 50)
        print(results['email'])


def main():
    """Main function to run the multi-agent system"""
    # Initialize with your Groq API key
    API_KEY = "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"
    
    try:
        # Create orchestrator
        orchestrator = MultiAgentOrchestrator(API_KEY)
        
        print("🤖 Multi-Agent AI System Initialized")
        print("Available commands:")
        print("1. 'analyze' - Run full analysis workflow")
        print("2. 'ask [question]' - Customer support query")
        print("3. 'quit' - Exit system")
        
        while True:
            user_input = input("\n💬 Enter command: ").strip().lower()
            
            if user_input == 'quit':
                print("👋 Goodbye!")
                break
            elif user_input == 'analyze':
                results = orchestrator.run_full_analysis()
                orchestrator.display_results(results)
            elif user_input.startswith('ask '):
                question = user_input[4:]  # Remove 'ask ' prefix
                answer = orchestrator.customer_support_query(question)
                print(f"\n🎧 Customer Support Response:")
                print("-" * 50)
                print(answer)
            else:
                print("❌ Invalid command. Use 'analyze', 'ask [question]', or 'quit'")
                
    except Exception as e:
        print(f"❌ System Error: {e}")


if __name__ == "__main__":
    main()