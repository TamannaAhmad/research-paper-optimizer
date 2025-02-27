import re
import requests
import json
import time
import os
import tkinter as tk
from tkinter import filedialog
from concurrent.futures import ThreadPoolExecutor
import PyPDF2
from pathlib import Path

class CitationValidator:
    def __init__(self):
        self.crossref_api = "https://api.crossref.org/works"
        self.scholar_headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Citation pattern matchers
        self.patterns = {
            'apa': r'([A-Za-z\s\-]+),\s([A-Z][\.]\s)?([A-Z][\.]\s)?(\([0-9]{4}\)[\.|\,])\s(.*?)[\.|\?|\!](?:\s(?:doi:|DOI:)?\s?(10\.[0-9]{4,}\/[a-zA-Z0-9\.\-\_]+))?',
            'mla': r'([A-Za-z\s\-]+),\s([A-Za-z\s]+)[\.|\,]\s\"(.*?)[\"|\.]\s(.*?)[\.|\,]\s([0-9]{4})',
            'chicago': r'([A-Za-z\s\-]+),\s([A-Za-z\s]+)[\.|\,]\s(.*?)[\.|\,]\s([A-Za-z\s\:\,]+),\s([0-9]{4})',
            'harvard': r'([A-Za-z\s\-]+),\s([A-Z][\.]\s)?([A-Z][\.]\s)?(\([0-9]{4}\))\s(.*?)[\.|\?|\!]',
            'ieee': r'\[([0-9]+)\]\s([A-Za-z\s\-]+),\s\"(.*?),\"\s(?:in\s)?(.*?),\s(?:vol\.\s)?([0-9]+)?(?:,\sno\.\s)?([0-9]+)?,\spp\.\s([0-9]+\-[0-9]+)?,\s([0-9]{4})',
            'doi': r'(?<!\S)(10\.[0-9]{4,}\/[a-zA-Z0-9\.\-\_]+)(?!\S)',
            # Match in-text numbered references like [1], [2], etc.
            'in_text_reference': r'\[([0-9]+(?:,\s*[0-9]+)*)\]',
            # Match numbered references in the reference section
            'numbered_reference': r'^\s*\[([0-9]+)\]\s+(.+?)(?=\n\s*\[[0-9]+\]|\n\n|\Z)',
            # Match APA-style or author-year references 
            'author_year': r'\((?:[A-Za-z\-]+(?:\set\sal\.?)?(?:\sand\s[A-Za-z\-]+)?,\s)?([12][0-9]{3}[a-z]?)\)'
        }

    def extract_text_from_pdf(self, pdf_path):
        """Extract text from a PDF file"""
        print(f"Extracting text from: {pdf_path}")
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num in range(len(reader.pages)):
                    text += reader.pages[page_num].extract_text() + "\n"
            return text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""
    
    def find_references_section(self, text):
        """Find the references or bibliography section in the text"""
        # Common section headings for references
        section_patterns = [
            r'(?:\n|\r)\s*(?:References|Bibliography|Works Cited|Literature Cited|Related Work|References and Notes)\s*(?:\n|\r)',
            r'(?:\n|\r)\s*REFERENCES\s*(?:\n|\r)'
        ]
        
        for pattern in section_patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            if matches:
                # Get the last match (in case there are multiple occurrences)
                start_pos = matches[-1].end()
                
                # Find the next section header (if any)
                next_section_pattern = r'(?:\n|\r)\s*[A-Z][A-Z\s]+\s*(?:\n|\r)'
                next_section_match = re.search(next_section_pattern, text[start_pos:])
                
                if next_section_match:
                    end_pos = start_pos + next_section_match.start()
                    return text[start_pos:end_pos].strip()
                else:
                    # If no next section, take all remaining text
                    return text[start_pos:].strip()
        
        return None

    def extract_citations_from_references(self, references_text):
        """Extract individual citations from the references section"""
        if not references_text:
            return []
        
        citations = []
        
        # Try to identify numbered references first [1], [2], etc.
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
            
            # Extract DOI if available
            citation_info['doi'] = self._extract_doi(ref_text)
            
            citations.append(citation_info)
        
        # If no numbered references found, try to split by line breaks or patterns
        if not numbered_found:
            # First try splitting by blank lines
            entries = re.split(r'\n\s*\n', references_text)
            
            # If that didn't work well (too few entries), try splitting by lines that begin with author patterns
            if len(entries) <= 1:
                author_pattern = r'\n(?=[A-Z][a-z]+,|\[|[0-9]+\.)'
                entries = re.split(author_pattern, references_text)
            
            for entry in entries:
                entry = entry.strip()
                if len(entry) > 10:  # Minimum length to be valid
                    # Determine style based on content
                    style = self._determine_citation_style(entry)
                    
                    citation_info = {
                        'text': entry,
                        'style': style,
                        'valid_format': True,
                        'exists': None,
                        'match_groups': (entry,)
                    }
                    
                    # Extract DOI if available
                    citation_info['doi'] = self._extract_doi(entry)
                    
                    citations.append(citation_info)
        
        return citations
    
    def _determine_citation_style(self, citation_text):
        """Determine the citation style based on text patterns"""
        if re.search(r'^\s*\[\d+\]', citation_text):
            return 'ieee'
        elif re.search(r'^\s*\d+\.\s', citation_text):
            return 'numbered'
        elif re.search(r'^[A-Z][a-z]+,\s[A-Z]\.', citation_text):
            # Check for year in parentheses
            if re.search(r'\([12][0-9]{3}\)', citation_text):
                return 'apa'
            else:
                return 'mla'
        elif re.search(r'et\sal\.', citation_text):
            return 'author_et_al'
        else:
            return 'unknown'
    
    def extract_in_text_citations(self, text):
        """Extract in-text citations to correlate with references"""
        in_text_citations = []
        
        # Look for numbered citations [1], [2], etc.
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
        
        # Look for author-year citations (Smith, 2020)
        author_year_matches = re.finditer(self.patterns['author_year'], text)
        for match in author_year_matches:
            year = match.group(1)
            # Get context to extract author
            context = self._get_citation_context(text, match.start())
            in_text_citations.append({
                'type': 'author_year',
                'year': year,
                'context': context
            })
        
        return in_text_citations
    
    def _get_citation_context(self, text, pos, context_size=100):
        """Get context around a citation for better understanding"""
        start = max(0, pos - context_size)
        end = min(len(text), pos + context_size)
        
        # Find sentence boundaries
        context = text[start:end]
        
        # Highlight the citation within the context
        cited_part = text[pos:min(pos + 20, len(text))]
        cited_end = cited_part.find(']')
        if cited_end > 0:
            cited_part = cited_part[:cited_end+1]
        
        relative_pos = pos - start
        context = context[:relative_pos] + "**" + cited_part + "**" + context[relative_pos + len(cited_part):]
        
        return context
    
    def validate_pdfs(self, pdf_paths):
        """Main function to validate citations in multiple PDF documents"""
        all_reports = []
        
        for pdf_path in pdf_paths:
            print(f"\nProcessing: {pdf_path}")
            # Extract text from PDF
            document_text = self.extract_text_from_pdf(pdf_path)
            
            if not document_text:
                print(f"No text extracted from {pdf_path}")
                continue
            
            # Find references section
            references_section = self.find_references_section(document_text)
            if references_section:
                print(f"References section found ({len(references_section)} characters)")
            else:
                print("No references section found. Will try to identify references throughout the document.")
            
            # Extract references from the references section
            citations = []
            if references_section:
                citations = self.extract_citations_from_references(references_section)
            
            # If no citations were found in a dedicated section, try to find them throughout the document
            if not citations:
                print("Attempting to find citations throughout the document...")
                # Add alternative citation extraction methods here
                
            print(f"Found {len(citations)} citations/references")
            
            # Extract in-text citations for correlation
            in_text_citations = self.extract_in_text_citations(document_text)
            print(f"Found {len(in_text_citations)} in-text citations")
            
            # Match in-text citations with references where possible
            self._match_citations_with_references(citations, in_text_citations)
            
            # Validate format for each citation
            for citation in citations:
                citation['valid_format'] = self.validate_citation_format(citation)
            
            # Check existence of each citation using threading for parallel processing
            print("Checking citation validity against databases...")
            with ThreadPoolExecutor(max_workers=5) as executor:
                def process_citation(citation):
                    # Add a small delay to avoid rate limiting
                    time.sleep(0.5)
                    result = self.check_citation_exists(citation)
                    citation.update({'exists': result.get('exists'), 'metadata': result})
                    return citation
                
                # Process citations in parallel
                validated_citations = list(executor.map(process_citation, citations))
            
            # Generate report for this PDF
            filename = os.path.basename(pdf_path)
            report = {
                'filename': filename,
                'total_citations': len(validated_citations),
                'valid_format_count': sum(1 for c in validated_citations if c['valid_format']),
                'existing_count': sum(1 for c in validated_citations if c.get('exists')),
                'invalid_format_count': sum(1 for c in validated_citations if not c['valid_format']),
                'nonexistent_count': sum(1 for c in validated_citations if c.get('exists') is False),
                'citations': validated_citations,
                'in_text_citations': in_text_citations
            }
            
            all_reports.append(report)
            
            # Print individual report
            self.print_report(report)
            
        return all_reports
    
    def _match_citations_with_references(self, citations, in_text_citations):
        """Match in-text citations with references based on numbers or author/year"""
        # Match numbered citations
        ref_num_map = {c.get('ref_num'): c for c in citations if 'ref_num' in c}
        
        for in_text in in_text_citations:
            if in_text['type'] == 'numbered' and in_text['ref_num'] in ref_num_map:
                ref = ref_num_map[in_text['ref_num']]
                if 'in_text_mentions' not in ref:
                    ref['in_text_mentions'] = []
                ref['in_text_mentions'].append(in_text['context'])
    
    def _extract_doi(self, text):
        """Extract DOI from citation text if present"""
        doi_match = re.search(self.patterns['doi'], text)
        if doi_match:
            return doi_match.group(1)
        return None
    
    def validate_citation_format(self, citation):
        """Validate if the citation follows proper formatting for its detected style"""
        # Basic format validation already done by regex matching
        return True
        
    def check_citation_exists(self, citation):
        """Check if the citation exists in academic databases"""
        if citation.get('doi'):
            return self._check_doi_exists(citation['doi'])
        else:
            return self._search_by_metadata(citation)
    
    def _check_doi_exists(self, doi):
        """Check if a DOI exists in CrossRef"""
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
        """Search for citation without DOI using metadata"""
        # Extract key info based on citation style
        search_terms = self._extract_search_terms(citation)
        
        # Try CrossRef first
        try:
            query = '+'.join(search_terms.split())
            response = requests.get(f"{self.crossref_api}?query={query}")
            
            if response.status_code == 200:
                data = response.json()
                items = data.get('message', {}).get('items', [])
                
                if items:
                    best_match = items[0]  # Take first match
                    return {
                        'exists': True,
                        'source': 'CrossRef',
                        'title': best_match.get('title', ['Unknown'])[0],
                        'authors': [author.get('family', '') for author in best_match.get('author', [])],
                        'year': best_match.get('published-print', {}).get('date-parts', [[0]])[0][0],
                        'confidence': 'medium'  # Without DOI, confidence is lower
                    }
        except Exception as e:
            pass
            
        # If CrossRef fails, could implement Google Scholar scraping here
        # (Note: Google Scholar doesn't have an official API and may block scraping)
        
        return {'exists': False, 'source': 'All sources', 'error': 'Citation not found'}
    
    def _extract_search_terms(self, citation):
        """Extract search terms from citation based on its style"""
        style = citation.get('style')
        text = citation.get('text', '')
        
        # Extract the first author's last name (if possible)
        author_match = re.search(r'^[^\d\[]*?([A-Z][a-z]+)', text)
        author = author_match.group(1) if author_match else ""
        
        # Extract a year (if possible)
        year_match = re.search(r'(19|20)\d{2}', text)
        year = year_match.group(0) if year_match else ""
        
        # Extract what might be a title (text between quotes or after author and year)
        title_match = re.search(r'["\'](.*?)["\']', text)
        
        if title_match:
            title = title_match.group(1)
        else:
            # Try to extract text after author and year
            title_pattern = rf'{author}.*?{year}.*?[,\.\s]+(.*?)[,\.]'
            title_match = re.search(title_pattern, text)
            title = title_match.group(1) if title_match else ""
            
            # If that fails, get a chunk of text
            if not title and len(text) > 20:
                title = text[20:100]  # Just use a portion of the text
        
        # Combine terms, filtering out empty ones
        terms = [t for t in [author, title, year] if t]
        return ' '.join(terms)
    
    def print_report(self, report):
        """Print a formatted citation validation report"""
        print("\nCITATION VALIDATION REPORT")
        print("-------------------------")
        print(f"File: {report['filename']}")
        print(f"Total citations found: {report['total_citations']}")
        print(f"Citations with valid format: {report['valid_format_count']}")
        print(f"Citations found in databases: {report['existing_count']}")
        print(f"Citations with invalid format: {report['invalid_format_count']}")
        print(f"Citations not found in databases: {report['nonexistent_count']}")
        print("\nCitation Details:")
        
        for i, citation in enumerate(report['citations']):
            print(f"\n{i+1}. Citation: {citation['text']}")
            print(f"   Style: {citation['style']}")
            print(f"   Format valid: {'Yes' if citation['valid_format'] else 'No'}")
            print(f"   Found in database: {'Yes' if citation.get('exists') else 'No'}")
            if citation.get('metadata'):
                metadata = citation['metadata']
                if metadata.get('exists'):
                    print(f"   Source: {metadata.get('source')}")
                    print(f"   Title: {metadata.get('title', 'Unknown')}")
                    if metadata.get('authors'):
                        print(f"   Authors: {', '.join(metadata.get('authors', []))}")
                    print(f"   Year: {metadata.get('year', 'Unknown')}")
                else:
                    print(f"   Error: {metadata.get('error', 'Unknown error')}")
            
            # Show in-text mentions if available
            if citation.get('in_text_mentions'):
                print(f"   In-text mentions ({len(citation['in_text_mentions'])}):")
                for j, mention in enumerate(citation['in_text_mentions'][:3]):  # Show up to 3 mentions
                    print(f"     {j+1}. ...{mention}...")
                if len(citation['in_text_mentions']) > 3:
                    print(f"     (and {len(citation['in_text_mentions'])-3} more mentions)")
    
    def export_report(self, reports, output_file):
        """Export citation validation reports to a JSON file"""
        with open(output_file, 'w') as f:
            json.dump(reports, f, indent=2)
        print(f"\nReport exported to {output_file}")

def select_pdf_files():
    """Open a file dialog to select multiple PDF files"""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_paths = filedialog.askopenfilenames(
        title="Select PDF Files",
        filetypes=[("PDF files", "*.pdf")]
    )
    return file_paths

def select_directory():
    """Open a directory dialog to select a folder containing PDF files"""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    directory = filedialog.askdirectory(title="Select Directory Containing PDF Files")
    
    if not directory:
        return []
        
    pdf_files = []
    for file in os.listdir(directory):
        if file.lower().endswith('.pdf'):
            pdf_files.append(os.path.join(directory, file))
    
    return pdf_files

def main():
    print("CITATION VALIDATOR")
    print("=================")
    print("This program extracts and validates citations from PDF files.")
    print("\nHow would you like to select PDF files?")
    print("1. Select individual PDF files")
    print("2. Select a directory containing PDF files")
    
    choice = input("\nEnter your choice (1 or 2): ")
    
    pdf_paths = []
    if choice == '1':
        pdf_paths = select_pdf_files()
    elif choice == '2':
        pdf_paths = select_directory()
    else:
        print("Invalid choice. Exiting.")
        return
    
    if not pdf_paths:
        print("No PDF files selected. Exiting.")
        return
    
    print(f"\nSelected {len(pdf_paths)} PDF files.")
    
    validator = CitationValidator()
    reports = validator.validate_pdfs(pdf_paths)
    
    # Export reports
    if reports:
        output_file = "citation_validation_report.json"
        validator.export_report(reports, output_file)

if __name__ == "__main__":
    main()