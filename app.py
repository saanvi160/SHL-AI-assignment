import sys

# Patch distutils if missing (Python 3.12 fix)
try:
    import distutils
except ImportError:
    import setuptools
    import types
    sys.modules['distutils'] = types.ModuleType("distutils")
    sys.modules['distutils.dir_util'] = types.ModuleType("distutils.dir_util")
    from setuptools._distutils.dir_util import copy_tree
    sys.modules['distutils.dir_util'].copy_tree = copy_tree

import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from typing import List, Dict, Tuple, Optional
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Set page configuration
st.set_page_config(
    page_title="SHL Assessment Recommendation System",
    page_icon="🧪",
    layout="wide"
)

# Initialize the embedding model
# @st.cache_resource
# def load_embedding_model():
#     return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


# Scrape and process SHL assessment data
@st.cache_data
def get_shl_assessments():
    try:
        # First, try to load from saved file if it exists
        if os.path.exists('shl_assessments.json'):
            with open('shl_assessments.json', 'r') as f:
                assessments = json.load(f)
                if assessments:  # Check if the list is not empty
                    return assessments
        
        # If file doesn't exist or is empty, scrape the data
        url = "https://www.shl.com/solutions/products/product-catalog/"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        assessments = []
        for item in soup.select('.product-item'):
            try:
                name = item.select_one('.product-title').text.strip()
                url = item.select_one('a')['href']
                
                details_dict = {}
                details = item.select('.detail')
                for detail in details:
                    # Extract labels and values
                    label_elem = detail.select_one('.detail-label')
                    value_elem = detail.select_one('.detail-value')
                    
                    if label_elem and value_elem:
                        label = label_elem.text.strip().replace(':', '')
                        value = value_elem.text.strip()
                        details_dict[label] = value
                
                # Extract information about remote testing and adaptive/IRT support
                remote_testing = 'Yes' if 'Remote Testing Support' in details_dict and details_dict['Remote Testing Support'] == 'Yes' else 'No'
                adaptive_irt = 'Yes' if 'Adaptive/IRT Support' in details_dict and details_dict['Adaptive/IRT Support'] == 'Yes' else 'No'
                
                # Extract duration
                duration = details_dict.get('Duration', 'Not specified')
                
                # Extract test type
                test_type = details_dict.get('Test Type', 'Not specified')
                
                # Check if this is an individual test solution
                if test_type != 'Not specified':
                    assessment = {
                        'name': name,
                        'url': url,
                        'remote_testing': remote_testing,
                        'adaptive_irt': adaptive_irt,
                        'duration': duration,
                        'test_type': test_type,
                        'details': details_dict
                    }
                    assessments.append(assessment)
            except Exception as e:
                continue
        
        # If web scraping failed or returned no results, use fallback data
        if not assessments:
            assessments = get_fallback_assessments()
        
        # Save to file for future use
        with open('shl_assessments.json', 'w') as f:
            json.dump(assessments, f)
            
        return assessments
    
    except Exception as e:
        st.error(f"Error scraping SHL assessments: {e}")
        # Return fallback data
        return get_fallback_assessments()

def get_fallback_assessments():
    """Provide fallback assessment data when scraping fails"""
    return [
        {
            'name': 'SHL Verify Interactive - Logical Reasoning',
            'url': 'https://www.shl.com/solutions/products/verify-interactive-logical/',
            'remote_testing': 'Yes',
            'adaptive_irt': 'No',
            'duration': '20 minutes',
            'test_type': 'Ability',
            'details': {
                'Duration': '20 minutes',
                'Test Type': 'Ability',
                'Remote Testing Support': 'Yes',
                'Adaptive/IRT Support': 'No'
            }
        },
        {
            'name': 'SHL Verify Interactive - Numerical Reasoning',
            'url': 'https://www.shl.com/solutions/products/verify-interactive-numerical/',
            'remote_testing': 'Yes',
            'adaptive_irt': 'No',
            'duration': '25 minutes',
            'test_type': 'Ability',
            'details': {
                'Duration': '25 minutes',
                'Test Type': 'Ability',
                'Remote Testing Support': 'Yes',
                'Adaptive/IRT Support': 'No'
            }
        },
        {
            'name': 'SHL Verify - Verbal Reasoning',
            'url': 'https://www.shl.com/solutions/products/verify-verbal/',
            'remote_testing': 'Yes', 
            'adaptive_irt': 'Yes',
            'duration': '18 minutes',
            'test_type': 'Ability',
            'details': {
                'Duration': '18 minutes',
                'Test Type': 'Ability',
                'Remote Testing Support': 'Yes',
                'Adaptive/IRT Support': 'Yes'
            }
        },
        {
            'name': 'SHL OPQ - Occupational Personality Questionnaire',
            'url': 'https://www.shl.com/solutions/products/opq/',
            'remote_testing': 'Yes',
            'adaptive_irt': 'No',
            'duration': '25 minutes',
            'test_type': 'Personality',
            'details': {
                'Duration': '25 minutes',
                'Test Type': 'Personality',
                'Remote Testing Support': 'Yes',
                'Adaptive/IRT Support': 'No'
            }
        },
        {
            'name': 'SHL MQ - Motivation Questionnaire',
            'url': 'https://www.shl.com/solutions/products/mq/',
            'remote_testing': 'Yes',
            'adaptive_irt': 'No',
            'duration': '25 minutes',
            'test_type': 'Motivation',
            'details': {
                'Duration': '25 minutes',
                'Test Type': 'Motivation',
                'Remote Testing Support': 'Yes',
                'Adaptive/IRT Support': 'No'
            }
        },
        {
            'name': 'SHL Scenarios - Customer Service',
            'url': 'https://www.shl.com/solutions/products/scenarios-customer-service/',
            'remote_testing': 'Yes',
            'adaptive_irt': 'No',
            'duration': '30 minutes',
            'test_type': 'Situational Judgment',
            'details': {
                'Duration': '30 minutes',
                'Test Type': 'Situational Judgment',
                'Remote Testing Support': 'Yes',
                'Adaptive/IRT Support': 'No'
            }
        },
        {
            'name': 'SHL Scenarios - Management',
            'url': 'https://www.shl.com/solutions/products/scenarios-management/',
            'remote_testing': 'Yes',
            'adaptive_irt': 'No',
            'duration': '30 minutes',
            'test_type': 'Situational Judgment',
            'details': {
                'Duration': '30 minutes', 
                'Test Type': 'Situational Judgment',
                'Remote Testing Support': 'Yes',
                'Adaptive/IRT Support': 'No'
            }
        },
        {
            'name': 'SHL Verify - Coding Simulation',
            'url': 'https://www.shl.com/solutions/products/verify-coding/',
            'remote_testing': 'Yes',
            'adaptive_irt': 'No',
            'duration': '60 minutes',
            'test_type': 'Skill',
            'details': {
                'Duration': '60 minutes',
                'Test Type': 'Skill',
                'Remote Testing Support': 'Yes',
                'Adaptive/IRT Support': 'No'
            }
        },
        {
            'name': 'SHL Verify - Microsoft Excel',
            'url': 'https://www.shl.com/solutions/products/verify-microsoft-excel/',
            'remote_testing': 'Yes',
            'adaptive_irt': 'No', 
            'duration': '30 minutes',
            'test_type': 'Skill',
            'details': {
                'Duration': '30 minutes',
                'Test Type': 'Skill',
                'Remote Testing Support': 'Yes',
                'Adaptive/IRT Support': 'No'
            }
        },
        {
            'name': 'SHL Verify - Data Analysis',
            'url': 'https://www.shl.com/solutions/products/verify-data-analysis/',
            'remote_testing': 'Yes',
            'adaptive_irt': 'No',
            'duration': '35 minutes',
            'test_type': 'Skill',
            'details': {
                'Duration': '35 minutes',
                'Test Type': 'Skill',
                'Remote Testing Support': 'Yes',
                'Adaptive/IRT Support': 'No'
            }
        }
    ]

# Function to generate embeddings for assessments
@st.cache_data
def generate_assessment_embeddings(assessments):
    if not assessments:
        return np.array([]).reshape(0, 384)  # Return empty array with correct shape
    
    texts = []
    for assessment in assessments:
        # Create a rich text representation of each assessment
        text = f"{assessment['name']}. {assessment['test_type']}. "
        for key, value in assessment['details'].items():
            text += f"{key}: {value}. "
        texts.append(text)
    
    # Generate embeddings
    if texts:
        embeddings = embedding_model.encode(texts)
        return embeddings
    else:
        # Return an empty array with the correct shape
        return np.array([]).reshape(0, 384)  # 384 is the dimension for all-MiniLM-L6-v2

# Function to extract text from a URL
def extract_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
            
        # Get text
        text = soup.get_text(separator=' ', strip=True)
        
        # Break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Drop blank lines
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    except Exception as e:
        st.error(f"Error extracting text from URL: {e}")
        return None

# Function to use Gemini to extract job requirements
def extract_job_requirements(text):
    model = genai.GenerativeModel('gemini-1.5-pro-001')
    
    prompt = f"""
    Please analyze the following job description and extract the key skills, requirements, 
    competencies, and traits needed for this role. Format the output as a concise, bulleted list 
    of requirements without any additional commentary.
    
    Job Description:
    {text[:10000]}  # Limit to first 10000 chars to stay within context window
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error extracting job requirements: {e}")
        return text  # Fallback to original text

# Function to rank assessments for a given query
def rank_assessments(query_text, assessments, assessment_embeddings):
    # Check if we have assessments to rank
    if not assessments or len(assessments) == 0:
        st.error("No assessments available to rank")
        return []
    
    # Check if we have embeddings
    if assessment_embeddings.size == 0:
        st.error("No assessment embeddings available")
        return []
    
    # Use Gemini to enhance the query
    model = genai.GenerativeModel('gemini-1.5-pro-001')
    
    prompt = f"""
    I'm looking for SHL assessments for a role with the following requirements:
    {query_text}
    
    Please convert this into a detailed search query that focuses on skills, abilities, and traits 
    that would be important to assess for this role. Include specific competencies, cognitive abilities,
    and personality traits.
    """
    try:
        response = model.generate_content(prompt)
        enhanced_query = response.text
    except Exception as e:
        st.warning(f"Error enhancing query: {e}. Using original query.")
        enhanced_query = query_text
    
    # Generate embedding for the enhanced query
    try:
        query_embedding = embedding_model.encode([enhanced_query])[0]
        
        # Calculate similarities
        similarities = cosine_similarity([query_embedding], assessment_embeddings)[0]
        
        # Get the indices of the top 10 most similar assessments
        top_indices = np.argsort(similarities)[::-1][:10]
        
        # Create a list of (assessment, similarity_score) tuples
        ranked_assessments = [(assessments[i], similarities[i]) for i in top_indices if similarities[i] > 0.1]
        
        # Use Gemini to rerank based on relevance to the job
        if ranked_assessments:
            assessment_summaries = "\n\n".join([
                f"Assessment {i+1}:\nName: {assessment['name']}\nType: {assessment['test_type']}\nDetails: {assessment['details']}"
                for i, (assessment, _) in enumerate(ranked_assessments[:15])  # Consider up to 15 for reranking
            ])
            
            rerank_prompt = f"""
            I have a job with these requirements:
            {query_text}
            
            Here are some potential SHL assessments:
            {assessment_summaries}
            
            Based on relevance to the job requirements, rank these assessments from most relevant to least relevant.
            For each assessment, give a short explanation of why it's relevant.
            Format your response as a list with assessment numbers followed by explanations.
            Example:
            1. Assessment 3 - Highly relevant because...
            2. Assessment 1 - Relevant because...
            """
            
            try:
                response = model.generate_content(rerank_prompt)
                reranking_text = response.text
                
                # Extract the reranked assessment numbers
                reranked_indices = []
                for line in reranking_text.split('\n'):
                    if re.match(r'^\d+\.?\s+Assessment\s+(\d+)', line):
                        match = re.search(r'Assessment\s+(\d+)', line)
                        if match:
                            index = int(match.group(1)) - 1
                            if 0 <= index < len(ranked_assessments):
                                reranked_indices.append(index)
                
                # Reorder the ranked_assessments based on the reranked indices
                if reranked_indices:
                    # Get unique indices (in case of duplicates)
                    unique_reranked = []
                    for i in reranked_indices:
                        if i not in unique_reranked:
                            unique_reranked.append(i)
                    
                    # Add any missing indices at the end
                    for i in range(len(ranked_assessments)):
                        if i not in unique_reranked:
                            unique_reranked.append(i)
                    
                    # Reorder the assessments
                    ranked_assessments = [ranked_assessments[i] for i in unique_reranked[:10]]
            
            except Exception as e:
                st.warning(f"Error during reranking: {e}. Using similarity-based ranking.")
        
        # Return up to 10 assessments
        return ranked_assessments[:10]
    
    except Exception as e:
        st.error(f"Error in ranking assessments: {e}")
        return []

# Main function
def main():
    st.title("🧪 SHL Assessment Recommendation System")
    
    st.write("""
    This system recommends SHL assessments based on job descriptions or requirements.
    You can input a job description directly, paste a URL, or enter specific requirements.
    """)
    
    # Input method selection
    input_method = st.radio(
        "Select input method:",
        ("Enter job description text", "Enter job description URL", "Enter specific requirements")
    )
    
    query_text = ""
    
    if input_method == "Enter job description text":
        query_text = st.text_area("Enter job description:", height=200)
        
    elif input_method == "Enter job description URL":
        url = st.text_input("Enter job description URL:")
        if url and st.button("Extract Text from URL"):
            with st.spinner("Extracting text from URL..."):
                text = extract_text_from_url(url)
                if text:
                    query_text = text
                    st.success("Text extracted successfully!")
                    with st.expander("View extracted text"):
                        st.text_area("Extracted text:", value=text, height=200, disabled=True)
    
    elif input_method == "Enter specific requirements":
        query_text = st.text_area("Enter job requirements:", height=150, 
                                  placeholder="Example: Looking for a software engineer with strong logical reasoning skills, problem-solving abilities, and good teamwork.")
    
    if query_text:
        # Process button
        if st.button("Recommend Assessments"):
            with st.spinner("Processing..."):
                # Load assessments
                assessments = get_shl_assessments()
                
                if not assessments:
                    st.error("Failed to load assessments. Please try again later.")
                else:
                    # If we have a full job description, extract the key requirements
                    if len(query_text.split()) > 50 and input_method != "Enter specific requirements":
                        with st.spinner("Extracting key job requirements..."):
                            query_text = extract_job_requirements(query_text)
                            with st.expander("View extracted job requirements"):
                                st.write(query_text)
                    
                    # Generate embeddings for assessments
                    assessment_embeddings = generate_assessment_embeddings(assessments)
                    
                    if assessment_embeddings.size == 0:
                        st.error("Failed to generate embeddings. Please try again later.")
                    else:
                        # Rank assessments for the query
                        ranked_assessments = rank_assessments(query_text, assessments, assessment_embeddings)
                        
                        # Display results
                        st.subheader("Recommended Assessments")
                        
                        if ranked_assessments:
                            # Create a DataFrame for display
                            results = []
                            for assessment, score in ranked_assessments:
                                results.append({
                                    "Assessment Name": assessment['name'],
                                    "Test Type": assessment['test_type'],
                                    "Remote Testing": assessment['remote_testing'],
                                    "Adaptive/IRT": assessment['adaptive_irt'],
                                    "Duration": assessment['duration'],
                                    "URL": assessment['url'],
                                    "Relevance Score": f"{score:.2f}"
                                })
                            
                            df = pd.DataFrame(results)
                            
                            # Display as HTML table with clickable links
                            st.markdown(
                                df.to_html(
                                    formatters={
                                        'URL': lambda x: f'<a href="{x}" target="_blank">{x}</a>'
                                    },
                                    escape=False,
                                    index=False
                                ),
                                unsafe_allow_html=True
                            )
                            
                            # Use Gemini to explain why these assessments are recommended
                            with st.expander("Why these assessments are recommended"):
                                try:
                                    model = genai.GenerativeModel('gemini-1.5-pro-001')
                                    assessments_summary = "\n".join([
                                        f"{i+1}. {assessment['name']} ({assessment['test_type']})"
                                        for i, (assessment, _) in enumerate(ranked_assessments[:5])
                                    ])
                                    
                                    explain_prompt = f"""
                                    For a role with these requirements:
                                    {query_text}
                                    
                                    I've recommended these SHL assessments:
                                    {assessments_summary}
                                    
                                    For each assessment, explain in 1-2 sentences why it's relevant for this role.
                                    Focus on how each assessment measures skills and traits important for the job.
                                    Format as a numbered list matching the assessments above.
                                    """
                                    
                                    response = model.generate_content(explain_prompt)
                                    st.write(response.text)
                                except Exception as e:
                                    st.error(f"Error generating explanation: {e}")
                        else:
                            st.warning("No relevant assessments found. Please try a different query.")
                    
    # Display sample queries
    with st.sidebar:
        st.subheader("Sample Queries")
        st.write("Click to use these sample queries:")
        
        sample_queries = [
            "Software Engineer with strong problem-solving skills and ability to work in a team",
            "Sales Manager with excellent communication and leadership skills",
            "Data Analyst with statistical knowledge and attention to detail",
            "Customer Service Representative with empathy and conflict resolution abilities"
        ]
        
        for query in sample_queries:
            if st.button(query, key=query):
                # This will be handled by the session state in Streamlit
                st.session_state.sample_query = query
                # Use st.rerun() instead of st.experimental_rerun()
                st.rerun()
    
    # Apply sample query if selected
    if hasattr(st.session_state, 'sample_query'):
        query = st.session_state.sample_query
        del st.session_state.sample_query
        
        if input_method == "Enter job description text" or input_method == "Enter specific requirements":
            st.text_area("Enter job description:" if input_method == "Enter job description text" else "Enter job requirements:", 
                        value=query, height=200, key="filled_query")

if __name__ == "__main__":
    main()
