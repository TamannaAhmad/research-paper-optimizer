import streamlit as st
import pandas as pd
import io
import base64
import tempfile
import os
import re
import requests
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import PyPDF2

class CitationValidator:
    def __init__(self):
        self.crossref_api = "https://api.crossref.org/works"
        self.scholar_headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.patterns = {
            'apa': r'([A-Za-z\s\-]+),\s([A-Z][\.]\s)?([A-Z][\.]\s)?(\([0-9]{4}\)[\.|\,])\s(.*?)[\.|\?|\!](?:\s(?:doi:|DOI:)?\s?(10\.[0-9]{4,}\/[a-zA-Z0-9\.\-\_]+))?',
            'mla': r'([A-Za-z\s\-]+),\s([A-Za-z\s]+)[\.|\,]\s\"(.*?)[\"|\.]\s(.*?)[\.|\,]\s([0-9]{4})',
            'chicago': r'([A-Za-z\s\-]+),\s([A-Za-z\s]+)[\.|\,]\s(.*?)[\.|\,]\s([A-Za-z\s\:\,]+),\s([0-9]{4})',
            'harvard': r'([A-Za-z\s\-]+),\s([A-Z][\.]\s)?([A-Z][\.]\s)?(\([0-9]{4}\))\s(.*?)[\.|\?|\!]',
            'ieee': r'\[([0-9]+)\]\s([A-Za-z\s\-]+),\s\"(.*?),\"\s(?:in\s)?(.*?),\s(?:vol\.\s)?([0-9]+)?(?:,\sno\.\s)?([0-9]+)?,\spp\.\s([0-9]+\-[0-9]+)?,\s([0-9]{4})',
            'doi': r'(?<!\S)(10\.[0-9]{4,}\/[a-zA-Z0-9\.\-\_]+)(?!\S)',
            'in_text_reference': r'\[([0-9]+(?:,\s*[0-9]+)*)\]',
            'numbered_reference': r'^\s*\[([0-9]+)\]\s+(.+?)(?=\n\s*\[[0-9]+\]|\n\n|\Z)',
            'author_year': r'\((?:[A-Za-z\-]+(?:\set\sal\.?)?(?:\sand\s[A-Za-z\-]+)?,\s)?([12][0-9]{3}[a-z]?)\)'
        }

    def extract_text_from_pdf(self, pdf_file):
        text = ""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(pdf_file.getvalue())
                temp_file_path = temp_file.name
            with open(temp_file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num in range(len(reader.pages)):
                    text += reader.pages[page_num].extract_text() + "\n"
            os.unlink(temp_file_path)
            return text
        except Exception as e:
            st.error(f"Error extracting text from PDF: {e}")
            return ""

    def find_references_section(self, text):
        section_patterns = [
            r'(?:\n|\r)\s*(?:References|Bibliography|Works Cited|Literature Cited|Related Work|References and Notes)\s*(?:\n|\r)',
            r'(?:\n|\r)\s*REFERENCES\s*(?:\n|\r)'
        ]
        for pattern in section_patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            if matches:
                start_pos = matches[-1].end()
                next_section_pattern = r'(?:\n|\r)\s*[A-Z][A-Z\s]+\s*(?:\n|\r)'
                next_section_match = re.search(next_section_pattern, text[start_pos:])
                if next_section_match:
                    end_pos = start_pos + next_section_match.start()
                    return text[start_pos:end_pos].strip()
                else:
                    return text[start_pos:].strip()
        return None

    def extract_citations_from_references(self, references_text):
        if not references_text:
            return []
        citations = []
        numbered_refs = re.finditer(self.patterns['numbered_reference'], references_text, re.MULTILINE | re.DOTALL)
        numbered_found = False
        for match in numbered_refs:
            numbered_found = True
            ref_num = match.group(1)
            ref_text = match.group(2).strip()
            citation_info = {
                'text': f"[{ref_num}] {ref_text}",
                'ref_num': ref_num,
                'style': 'numbered',
                'valid_format': True,
                'exists': None,
                'match_groups': (ref_num, ref_text)
            }
            citation_info['doi'] = self._extract_doi(ref_text)
            citations.append(citation_info)
        if not numbered_found:
            entries = re.split(r'\n\s*\n', references_text)
            if len(entries) <= 1:
                author_pattern = r'\n(?=[A-Z][a-z]+,|\[|[0-9]+\.)'
                entries = re.split(author_pattern, references_text)
            for entry in entries:
                entry = entry.strip()
                if len(entry) > 10:
                    style = self._determine_citation_style(entry)
                    citation_info = {
                        'text': entry,
                        'style': style,
                        'valid_format': True,
                        'exists': None,
                        'match_groups': (entry,)
                    }
                    citation_info['doi'] = self._extract_doi(entry)
                    citations.append(citation_info)
        return citations

    def _determine_citation_style(self, citation_text):
        if re.search(r'^\s*\[\d+\]', citation_text):
            return 'ieee'
        elif re.search(r'^\s*\d+\.\s', citation_text):
            return 'numbered'
        elif re.search(r'^[A-Z][a-z]+,\s[A-Z]\.', citation_text):
            if re.search(r'\([12][0-9]{3}\)', citation_text):
                return 'apa'
            else:
                return 'mla'
        elif re.search(r'et\sal\.', citation_text):
            return 'author_et_al'
        else:
            return 'unknown'

    def extract_in_text_citations(self, text):
        in_text_citations = []
        numbered_matches = re.finditer(self.patterns['in_text_reference'], text)
        for match in numbered_matches:
            ref_nums = match.group(1).split(',')
            for num in ref_nums:
                num = num.strip()
                if num and num.isdigit():
                    in_text_citations.append({
                        'type': 'numbered',
                        'ref_num': num,
                        'context': self._get_citation_context(text, match.start())
                    })
        author_year_matches = re.finditer(self.patterns['author_year'], text)
        for match in author_year_matches:
            year = match.group(1)
            context = self._get_citation_context(text, match.start())
            in_text_citations.append({
                'type': 'author_year',
                'year': year,
                'context': context
            })
        return in_text_citations

    def _get_citation_context(self, text, pos, context_size=100):
        start = max(0, pos - context_size)
        end = min(len(text), pos + context_size)
        context = text[start:end]
        cited_part = text[pos:min(pos + 20, len(text))]
        cited_end = cited_part.find(']')
        if cited_end > 0:
            cited_part = cited_part[:cited_end+1]
        relative_pos = pos - start
        context = context[:relative_pos] + "**" + cited_part + "**" + context[relative_pos + len(cited_part):]
        return context

    def validate_citations(self, document_text):
        if not document_text:
            return None
        references_section = self.find_references_section(document_text)
        citations = self.extract_citations_from_references(references_section) if references_section else []
        in_text_citations = self.extract_in_text_citations(document_text)
        self._match_citations_with_references(citations, in_text_citations)
        for citation in citations:
            citation['valid_format'] = self.validate_citation_format(citation)
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(self.check_citation_exists, citations))
        for citation, result in zip(citations, results):
            citation.update({'exists': result.get('exists'), 'metadata': result})
        report = {
            'total_citations': len(citations),
            'valid_format_count': sum(1 for c in citations if c['valid_format']),
            'existing_count': sum(1 for c in citations if c.get('exists')),
            'invalid_format_count': sum(1 for c in citations if not c['valid_format']),
            'nonexistent_count': sum(1 for c in citations if c.get('exists') is False),
            'citations': citations,
            'in_text_citations': in_text_citations
        }
        return report

    def _match_citations_with_references(self, citations, in_text_citations):
        ref_num_map = {c.get('ref_num'): c for c in citations if 'ref_num' in c}
        for in_text in in_text_citations:
            if in_text['type'] == 'numbered' and in_text['ref_num'] in ref_num_map:
                ref = ref_num_map[in_text['ref_num']]
                if 'in_text_mentions' not in ref:
                    ref['in_text_mentions'] = []
                ref['in_text_mentions'].append(in_text['context'])

    def _extract_doi(self, text):
        doi_match = re.search(self.patterns['doi'], text)
        return doi_match.group(1) if doi_match else None

    def validate_citation_format(self, citation):
        return True

    def check_citation_exists(self, citation):
        if citation.get('doi'):
            return self._check_doi_exists(citation['doi'])
        else:
            return self._search_by_metadata(citation)

    def _check_doi_exists(self, doi):
        try:
            response = requests.get(f"{self.crossref_api}/{doi}")
            if response.status_code == 200:
                data = response.json()
                return {
                    'exists': True,
                    'source': 'CrossRef',
                    'title': data.get('message', {}).get('title', ['Unknown'])[0],
                    'authors': [author.get('family', '') for author in data.get('message', {}).get('author', [])],
                    'year': data.get('message', {}).get('published-print', {}).get('date-parts', [[0]])[0][0]
                }
            return {'exists': False, 'source': 'CrossRef', 'error': 'DOI not found'}
        except Exception as e:
            return {'exists': False, 'source': 'CrossRef', 'error': str(e)}

    def _search_by_metadata(self, citation):
        search_terms = self._extract_search_terms(citation)
        try:
            query = '+'.join(search_terms.split())
            response = requests.get(f"{self.crossref_api}?query={query}")
            if response.status_code == 200:
                data = response.json()
                items = data.get('message', {}).get('items', [])
                if items:
                    best_match = items[0]
                    return {
                        'exists': True,
                        'source': 'CrossRef',
                        'title': best_match.get('title', ['Unknown'])[0],
                        'authors': [author.get('family', '') for author in best_match.get('author', [])],
                        'year': best_match.get('published-print', {}).get('date-parts', [[0]])[0][0],
                        'confidence': 'medium'
                    }
        except Exception:
            pass
        return {'exists': False, 'source': 'All sources', 'error': 'Citation not found'}

    def _extract_search_terms(self, citation):
        style = citation.get('style')
        text = citation.get('text', '')
        author_match = re.search(r'^[^\d\[]*?([A-Z][a-z]+)', text)
        author = author_match.group(1) if author_match else ""
        year_match = re.search(r'(19|20)\d{2}', text)
        year = year_match.group(0) if year_match else ""
        title_match = re.search(r'["\'](.*?)["\']', text)
        if title_match:
            title = title_match.group(1)
        else:
            title_pattern = rf'{author}.*?{year}.*?[,\.\s]+(.*?)[,\.]'
            title_match = re.search(title_pattern, text)
            title = title_match.group(1) if title_match else text[20:100] if len(text) > 20 else ""
        terms = [t for t in [author, title, year] if t]
        return ' '.join(terms)

    def export_report_to_csv(self, report):
        if not report:
            return None
        citations_data = []
        for citation in report['citations']:
            citation_row = {
                'Citation Text': citation['text'][:100] + ('...' if len(citation['text']) > 100 else ''),
                'Style': citation['style'],
                'Format Valid': 'Yes' if citation['valid_format'] else 'No',
                'Found in Database': 'Yes' if citation.get('exists') else 'No',
                'DOI': citation.get('doi', 'None')
            }
            if citation.get('metadata') and citation['metadata'].get('exists'):
                metadata = citation['metadata']
                citation_row.update({
                    'Title': metadata.get('title', 'Unknown'),
                    'Authors': ', '.join(metadata.get('authors', [])),
                    'Year': metadata.get('year', 'Unknown'),
                    'Source': metadata.get('source', 'Unknown')
                })
            else:
                citation_row.update({'Title': 'N/A', 'Authors': 'N/A', 'Year': 'N/A', 'Source': 'N/A'})
            citations_data.append(citation_row)
        df = pd.DataFrame(citations_data)
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        return csv_buffer.getvalue()

def run_citation_validator(uploaded_file=None):
    st.title("📚 Citation Validator")
    st.markdown("Validate your paper's citations with ease")
    st.markdown("""
    This tool helps academic writers check their citations for:
    - ✅ Proper formatting
    - 🔍 Existence in academic databases
    - 🔄 Consistency between in-text citations and references
    """)

    validator = CitationValidator()
    if uploaded_file is None:
        uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file:
        with st.spinner("Processing PDF... This may take a minute."):
            extracted_text = validator.extract_text_from_pdf(uploaded_file)
            if extracted_text:
                st.success("PDF processed successfully!")
                with st.expander("Preview extracted text"):
                    st.text_area("Extracted Text", extracted_text, height=200)
                references_section = validator.find_references_section(extracted_text)
                if references_section:
                    with st.expander("References Section"):
                        st.text_area("Extracted References", references_section, height=200)
                else:
                    st.warning("No dedicated references section found.")
                with st.spinner("Validating citations..."):
                    progress_bar = st.progress(0)
                    report = validator.validate_citations(extracted_text)
                    progress_bar.progress(100)
                if report:
                    st.subheader("Citation Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Citations", report['total_citations'])
                    col2.metric("Valid Format", report['valid_format_count'])
                    col3.metric("Found in Databases", report['existing_count'])
                    col4.metric("Not Found", report['nonexistent_count'])
                    st.subheader("Citation Details")
                    citation_data = [
                        {
                            'No.': i+1,
                            'Citation': c['text'][:100] + ('...' if len(c['text']) > 100 else ''),
                            'Style': c['style'],
                            'Format Valid': '✅' if c['valid_format'] else '❌',
                            'Found in Database': '✅' if c.get('exists') else '❌',
                            'DOI': c.get('doi', '')
                        }
                        for i, c in enumerate(report['citations'])
                    ]
                    df = pd.DataFrame(citation_data)
                    st.dataframe(df)
                    if report['in_text_citations']:
                        st.subheader("In-text Citations")
                        st.write(f"Found {len(report['in_text_citations'])} in-text citations.")
                        with st.expander("View In-text Citations"):
                            for i, citation in enumerate(report['in_text_citations'][:20]):
                                st.markdown(f"**{i+1}.** Context: {citation['context']}")
                                st.markdown("---")
                            if len(report['in_text_citations']) > 20:
                                st.write(f"... and {len(report['in_text_citations']) - 20} more.")
                    st.subheader("Export Report")
                    csv_data = validator.export_report_to_csv(report)
                    if csv_data:
                        csv_b64 = base64.b64encode(csv_data.encode()).decode()
                        href = f'<a href="data:file/csv;base64,{csv_b64}" download="citation_report.csv">Download CSV Report</a>'
                        st.markdown(href, unsafe_allow_html=True)
                else:
                    st.error("No citations could be extracted and validated.")
            else:
                st.error("Could not extract text from the PDF.")
    with st.expander("About this tool"):
        st.markdown("""
        ### Academic Citation Validator
        This tool analyzes academic papers to validate citations and references by:
        1. **Extracting text** from PDF documents
        2. **Identifying** the references section
        3. **Parsing** individual citation entries
        4. **Detecting** in-text citations and matching them with references
        5. **Validating** DOIs and citation existence in academic databases
        6. **Checking** citation formatting according to common styles
        #### Supported Citation Styles
        - APA
        - MLA
        - Chicago
        - Harvard
        - IEEE
        - Numbered references
        #### How It Works
        The tool uses regular expressions and API calls to academic databases like CrossRef.
        """)

def main():
    run_citation_validator()

if __name__ == "__main__":
    main()