# Import necessary libraries
import os
import requests
import json
from typing import Dict
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from IPython.display import Markdown, display, update_display
from transformers import pipeline, AutoTokenizer
from urllib.parse import urljoin
import streamlit as st
from huggingface_hub import login, InferenceClient

load_dotenv()

hf_token = os.environ['HF_TOKEN']
login(hf_token, add_to_git_credential=True)
MODEL = "meta-llama/Llama-3.1-8B-Instruct"

client = InferenceClient(api_key= hf_token)

class Website:

    def __init__(self, url: str):
        """
        Initialize the Website object with a given URL.
        
        :param url: The URL of the website to scrape
        """
        self.url = url
        self.title, self.text, self.links = self._scrape_website()

    def _scrape_website(self) -> tuple:

        response = requests.get(self.url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        title = soup.title.string if soup.title else "No title found"
        
        if soup.body:
            for tag in soup.body(["script", "style", "img", "input"]):
                tag.decompose()  
            text = soup.body.get_text(separator="\n", strip=True)
        else:
            text = ""
        
        links = [link.get('href') for link in soup.find_all('a') if link.get('href')]
        
        return title, text, links

    def get_contents(self) -> str:

        return f"Webpage Title:\n{self.title}\nWebpage Contents:\n{self.text}\n\n"

class LinkAnalyzer:

    LINK_SYSTEM_PROMPT = """
    You are provided with a list of links found on a webpage. Your task is to first categorize each link into one of the following categories:
    - about page
    - careers page
    - terms of service
    - privacy policy
    - contact page
    - other (please specify).

    Once the links are categorized, please choose which links are most relevant to include in a brochure about the company. 
    The brochure should only include links such as About pages, Careers pages, or Company Overview pages. Exclude any links related to Terms of Service, Privacy Policy, or email addresses.

    Respond in the following JSON format:
    {
        "categorized_links": [
            {"category": "about page", "url": "https://full.url/about"},
            {"category": "careers page", "url": "https://full.url/careers"},
            {"category": "terms of service", "url": "https://full.url/terms"},
            {"category": "privacy policy", "url": "https://full.url/privacy"},
            {"category": "other", "specify": "contact page", "url": "https://full.url/contact"}
        ],
        "brochure_links": [
            {"type": "about page", "url": "https://full.url/about"},
            {"type": "careers page", "url": "https://full.url/careers"}
        ]
    }

    Please find the links below and proceed with the task:

    Links (some may be relative links):
    [INSERT LINK LIST HERE]
    """

    @staticmethod
    def get_links(website: Website) -> Dict:

        user_prompt = f"Here is the list of links on the website of {website.url} - "
        user_prompt += "please decide which of these are relevant web links for a brochure about the company, respond with the full https URL in JSON format. \
    Do not include Terms of Service, Privacy, email links.\n"
        user_prompt += "Links (some might be relative links):\n"
        user_prompt += "\n".join(website.links)

        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": LinkAnalyzer.LINK_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(completion.choices[0].message.content)

class BrochureGenerator:

    SYSTEM_PROMPT = """
    You are an assistant that analyzes the contents of several relevant pages from a company website 
    and creates a brochure about the company for prospective customers, investors and recruits. Respond in markdown.
    Include details of company culture, customers and careers/jobs if you have the information.
    Structure the brochure to include specific sections as follows:
    About Us
    What we do
    How We Do It
    Where We Do It
    Our People
    Our Culture
    Connect with Us.
    Please provide two versions of the brochure, the first in English, the second in Spanish. The contents of the brochure are to be the same for both languages.
    """

    @staticmethod
    def get_all_details(url: str) -> str:

        result = "Landing page:\n"
        website = Website(url)
        result += website.get_contents()

        links = LinkAnalyzer.get_links(website)
        brochure_links = links.get('brochure_links', [])
        print("Found Brochure links:", brochure_links)

        for link in brochure_links:
            result += f"\n\n{link['type']}:\n"
            full_url = urljoin(url, link["url"])
            result += Website(full_url).get_contents()

        return result

    @staticmethod
    def get_brochure_user_prompt(company_name: str, url: str) -> str:
        
        user_prompt = f"You are looking at a company called: {company_name}\n"
        user_prompt += f"Here are the contents of its landing page and other relevant pages; use this information to build a short brochure of the company in markdown.\n"
        user_prompt += BrochureGenerator.get_all_details(url)
        return user_prompt[:20_000]  

    @staticmethod
    def stream_brochure(company_name: str, url: str):

        stream = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": BrochureGenerator.SYSTEM_PROMPT},
                {"role": "user", "content": BrochureGenerator.get_brochure_user_prompt(company_name, url)}
            ],
            stream=True
        )

        response = ""
        display_handle = display(Markdown(""), display_id=True)
        for chunk in stream:
            response += chunk.choices[0].delta.content or ''
            response = response.replace("```", "").replace("markdown", "")
            update_display(Markdown(response), display_id=display_handle.display_id)

st.title("Website Brochure Generator")

name = st.text_input("Enter the Name of the Company:", "")
url = st.text_input("Enter the URL of the website:", "")

if st.button("Generate Brochure"):
    if url:
        try:
            brochure_content = BrochureGenerator.stream_brochure(name, url)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.error("Please enter a valid URL.")
