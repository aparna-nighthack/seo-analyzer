import os
import re
import asyncio
from datetime import datetime
from urllib.parse import urlparse

import aiohttp
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import streamlit as st
from pymongo import MongoClient
from pymongo.errors import PyMongoError

# LangChain imports
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

class UsageTracker:
    def __init__(self):
        """Initialize MongoDB connection"""
        mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
        self.client = MongoClient(mongodb_uri)
        self.db = self.client["seo_analyzer"]
        self.usage_collection = self.db["organization_usage"]
        self.valid_orgs_collection = self.db["valid_organizations"]
        
        # Create indexes for faster queries
        self.usage_collection.create_index("org_id", unique=True)
        self.usage_collection.create_index("last_used")
        self.valid_orgs_collection.create_index("org_id", unique=True)
        
        # Initialize with default organizations if empty
        if self.valid_orgs_collection.count_documents({}) == 0:
            self.initialize_valid_organizations()

    def initialize_valid_organizations(self):
        """Seed the database with initial valid organizations"""
        default_orgs = ["K5$t&xW@"]  # Example org ID
        try:
            self.valid_orgs_collection.insert_many(
                [{"org_id": org_id} for org_id in default_orgs]
            )
        except PyMongoError as e:
            st.error(f"Error initializing valid organizations: {e}")

    def is_valid_organization(self, org_id: str) -> bool:
        """Check if organization ID exists in valid organizations"""
        try:
            return bool(self.valid_orgs_collection.find_one({"org_id": org_id.strip()}))
        except PyMongoError as e:
            st.error(f"MongoDB error checking valid orgs: {e}")
            return False

    def check_usage(self, org_id: str) -> dict:
        """Check organization's usage status"""
        try:
            org_record = self.usage_collection.find_one({"org_id": org_id})
            if org_record:
                remaining = max(0, 20 - org_record.get("usage_count", 0))
                return {
                    "exists": True,
                    "remaining": remaining,
                    "total_used": org_record.get("usage_count", 0),
                    "last_used": org_record.get("last_used")
                }
            return {
                "exists": False,
                "remaining": 20,
                "total_used": 0,
                "last_used": None
            }
        except PyMongoError as e:
            return {"error": str(e)}

    def increment_usage(self, org_id: str) -> bool:
        """Increment usage count for organization"""
        try:
            result = self.usage_collection.update_one(
                {"org_id": org_id},
                {
                    "$inc": {"usage_count": 1},
                    "$set": {"last_used": datetime.now()},
                    "$setOnInsert": {"created_at": datetime.now()}
                },
                upsert=True
            )
            return result.acknowledged
        except PyMongoError as e:
            st.error(f"MongoDB error: {e}")
            return False

    def close(self):
        """Close MongoDB connection"""
        self.client.close()

class SEOAnalyzer:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))
        self.setup_tools()
        self.setup_agent()
        self.setup_prompts()
        self.output_dir = "seo_reports"
        os.makedirs(self.output_dir, exist_ok=True)
        self.usage_tracker = UsageTracker()

    def validate_url(self, url: str) -> str:
        """Ensure URL has proper scheme and format"""
        url = url.strip()
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        parsed = urlparse(url)
        if not parsed.netloc:
            raise ValueError("Invalid URL format")
        return url

    async def fetch_with_headers(self, url: str) -> str:
        """Custom fetch function with headers"""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(url) as response:
                return await response.text()

    def setup_tools(self):
        """Configure tools for content extraction and analysis"""
        async def extract_content(url: str) -> str:
            try:
                validated_url = self.validate_url(url)
                html = await self.fetch_with_headers(validated_url)
                soup = BeautifulSoup(html, 'html.parser')
                
                for element in soup(['header', 'footer', 'nav', 'script', 'style']):
                    element.decompose()
                
                text = soup.get_text(separator=' ', strip=True)
                return text[:15000] if text else ""
            except Exception as e:
                return f"Extraction failed: {str(e)}"

        self.tools = [
            Tool(
                name="extract_website_content",
                func=lambda x: asyncio.run(extract_content(x)),
                description="Extracts main content from a website URL",
                coroutine=extract_content
            ),
            Tool(
                name="search_competitors",
                func=TavilySearchResults(k=3).run,
                description="Finds competitor websites for keyword analysis"
            )
        ]

    def setup_prompts(self):
        """Configure enhanced prompt templates for SEO tasks"""
        self.keyword_prompt = ChatPromptTemplate.from_template("""
        You are a professional SEO analyst. Extract keywords following these rules:
        1. Primary Keywords: 1-3 word phrases with high commercial intent
        2. Secondary Keywords: Longer-tail supporting phrases
        3. Include location modifiers if relevant
        4. No duplicates
        5. Prioritize treatment/service-specific terms
        
        Content: {content}
        
        Output EXACTLY in this format:
        Primary Keywords: [comma separated list]
        Secondary Keywords: [comma separated list]
        """)

        self.title_prompt = ChatPromptTemplate.from_template("""
        Create optimized title tag following:
        1. MAX 120 characters (current: {current_length})
        2. Include: {primary_keywords}
        3. Location at end if present
        4. Add business name if <100 chars
        5. NO pipes/slashes/symbols
        6. Natural flow
        
        Current Title: {current_title}
        Page Content: {content}
        
        Optimized Title:
        """)

        self.meta_prompt = ChatPromptTemplate.from_template("""
        Create meta description following:
        1. MAX 160 characters (current: {current_length})
        2. INCLUDE: {keywords}
        3. Engaging ad copy style
        4. CLEAR CTA at end
        5. Active voice ONLY
        
        Current Description: {current_meta}
        Page Content: {content}
        
        Optimized Meta:
        """)

    def setup_agent(self):
        """Initialize the analysis agent"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a professional SEO analyst following strict guidelines."),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        self.agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )

    async def analyze_page(self, url: str) -> dict:
        """Generate complete SEO report for a URL"""
        st.info(f"🔍 Analyzing {url}...")
        
        try:
            content = await self.tools[0].coroutine(url)
            if "Extraction failed" in content:
                raise ValueError(content)
                
            html = await self.fetch_with_headers(url)
            current_title = self.extract_meta_tag(html, "title")
            current_meta = self.extract_meta_tag(html, "meta")
            
            keyword_chain = self.keyword_prompt | self.llm | StrOutputParser()
            keyword_result = await keyword_chain.ainvoke({"content": content})
            primary_keywords, secondary_keywords = self.parse_keywords(keyword_result)
            
            title_chain = self.title_prompt | self.llm | StrOutputParser()
            optimized_title = await title_chain.ainvoke({
                "primary_keywords": primary_keywords,
                "current_title": current_title,
                "current_length": len(current_title),
                "content": content
            })
            
            meta_chain = self.meta_prompt | self.llm | StrOutputParser()
            optimized_meta = await meta_chain.ainvoke({
                "keywords": f"{primary_keywords}, {secondary_keywords}",
                "current_meta": current_meta,
                "current_length": len(current_meta),
                "content": content
            })
            
            competitor_query = f"top {primary_keywords.split(',')[0]} websites"
            competitors = self.tools[1].run(competitor_query)
            
            return {
                "url": url,
                "current_title": current_title,
                "optimized_title": optimized_title,
                "current_meta": current_meta,
                "optimized_meta": optimized_meta,
                "primary_keywords": primary_keywords,
                "secondary_keywords": secondary_keywords,
                "competitor_analysis": competitors[:3],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "url": url,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

    @staticmethod
    def extract_meta_tag(html: str, tag: str) -> str:
        """Extract title or meta description from HTML"""
        soup = BeautifulSoup(html, 'html.parser')
        if tag == "title":
            title_tag = soup.find('title')
            return title_tag.get_text().strip() if title_tag else "Not found"
        meta_tag = soup.find('meta', attrs={'name': 'description'})
        return meta_tag['content'].strip() if meta_tag else "Not found"

    @staticmethod
    def parse_keywords(keyword_result: str) -> tuple:
        """Extract keywords from analysis result"""
        try:
            primary = keyword_result.split("Primary Keywords:")[1].split("Secondary Keywords:")[0].strip()
            secondary = keyword_result.split("Secondary Keywords:")[1].strip()
            primary = re.sub(r'[\*\-\n]', '', primary).strip()
            secondary = re.sub(r'[\*\-\n]', '', secondary).strip()
            return primary, secondary
        except Exception:
            return "Keyword extraction failed", ""

    def save_to_markdown(self, analysis: dict):
        """Save results to markdown file"""
        filename = f"SEO_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as md_file:
            md_file.write(f"# SEO Analysis Report\n*Generated on {analysis.get('timestamp', '')}*\n\n")
            md_file.write(f"## URL\n{analysis.get('url', '')}\n\n")
            md_file.write(f"## Current Title Tag\n{analysis.get('current_title', '')}\n")
            md_file.write(f"*Length: {len(analysis.get('current_title', ''))} characters*\n\n")
            md_file.write(f"## Recommended Title Tag\n{analysis.get('optimized_title', '')}\n")
            md_file.write(f"*Length: {len(analysis.get('optimized_title', ''))} characters* ")
            md_file.write(f"{'✅' if len(analysis.get('optimized_title', '')) <= 120 else '⚠️'}\n\n")
            md_file.write(f"## Current Meta Description\n{analysis.get('current_meta', '')}\n")
            md_file.write(f"*Length: {len(analysis.get('current_meta', ''))} characters*\n\n")
            md_file.write(f"## Recommended Meta Description\n{analysis.get('optimized_meta', '')}\n")
            md_file.write(f"*Length: {len(analysis.get('optimized_meta', ''))} characters* ")
            md_file.write(f"{'✅' if len(analysis.get('optimized_meta', '')) <= 160 else '⚠️'}\n\n")
            md_file.write(f"## Primary Keywords\n{analysis.get('primary_keywords', '')}\n\n")
            md_file.write(f"## Secondary Keywords\n{analysis.get('secondary_keywords', '')}\n\n")
            md_file.write(f"## Top Competitors\n")
            for i, competitor in enumerate(analysis.get('competitor_analysis', [])[:3], 1):
                md_file.write(f"{i}. [{competitor.get('url', '')}]({competitor.get('url', '')})\n")

    def generate_report(self, analysis: dict) -> str:
        """Format the complete SEO report with validation"""
        if "error" in analysis:
            return f"❌ Error analyzing {analysis['url']}:\n{analysis['error']}"
            
        title_len = len(analysis.get('optimized_title', ''))
        meta_len = len(analysis.get('optimized_meta', ''))
        
        report = f"""
🚀 **SEO Optimization Report**  
**URL:** {analysis['url']}  
**Timestamp:** {analysis['timestamp']}

### Keyword Analysis
- **Primary Keywords:** {analysis.get('primary_keywords', '')}
- **Secondary Keywords:** {analysis.get('secondary_keywords', '')}

### Title Tag
- **Current Title:** {analysis.get('current_title', '')} ({len(analysis.get('current_title', ''))} chars)
- **Optimized Title:** {analysis.get('optimized_title', '')} ({title_len} chars) {"⚠️ Too long!" if title_len > 120 else "✅"}

### Meta Description
- **Current Meta:** {analysis.get('current_meta', '')} ({len(analysis.get('current_meta', ''))} chars)
- **Optimized Meta:** {analysis.get('optimized_meta', '')} ({meta_len} chars) {"⚠️ Too long!" if meta_len > 160 else "✅"}
"""
        return report

async def generate_seo_report(url: str, org_id: str) -> str:
    """Generate SEO report with usage tracking"""
    if not org_id.strip():
        return "❌ Organization ID is required"
    
    analyzer = SEOAnalyzer()
    
    # Check if organization is valid
    if not analyzer.usage_tracker.is_valid_organization(org_id):
        analyzer.usage_tracker.close()
        return "❌ Invalid Organization ID. Access denied."
    
    # Check usage limits
    usage = analyzer.usage_tracker.check_usage(org_id.strip())
    if "error" in usage:
        analyzer.usage_tracker.close()
        return f"❌ Error checking usage: {usage['error']}"
    
    if usage["remaining"] <= 0:
        analyzer.usage_tracker.close()
        last_used = usage["last_used"].strftime("%Y-%m-%d %H:%M") if usage["last_used"] else "never"
        return f"❌ Usage limit exceeded for organization {org_id}.\n\n• Total used: {usage['total_used']}/20\n• Last used: {last_used}"
    
    try:
        analysis = await analyzer.analyze_page(url)
        report = analyzer.generate_report(analysis)
        analyzer.save_to_markdown(analysis)
        
        # Update usage only on success
        if "error" not in analysis:
            if not analyzer.usage_tracker.increment_usage(org_id.strip()):
                report += "\n\n⚠️ Could not update usage tracking"
        
        report += f"\n\n🔢 Usage: {usage['total_used'] + 1}/20 reports"
        return report
    except Exception as e:
        return f"❌ Error generating report: {str(e)}"
    finally:
        analyzer.usage_tracker.close()

def main():
    st.set_page_config(
        page_title="SEO Analyzer",
        page_icon="🔍",
        layout="centered"
    )
    
    st.title("🔍 SEO Analyzer")
    st.caption("Generate SEO reports with usage tracking (20 reports per organization)")
    
    url = st.text_input("Website URL", placeholder="https://example.com")
    org_id = st.text_input("Organization ID", placeholder="Enter your organization ID")
    
    if st.button("Analyze", type="primary"):
        if not url or not org_id:
            st.error("Please provide both URL and Organization ID!")
        else:
            with st.spinner("Generating SEO report..."):
                report = asyncio.run(generate_seo_report(url, org_id))
                st.markdown(report)
                
                # Add download button
                st.download_button(
                    label="Download Report (Markdown)",
                    data=report,
                    file_name=f"seo_report_{datetime.now().strftime('%Y%m%d')}.md",
                    mime="text/markdown"
                )

if __name__ == "__main__":
    main()