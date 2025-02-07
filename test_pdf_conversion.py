from PyPDF2 import PdfReader

def test_pdf_conversion():
    # Specify the path to your PDF file
    pdf_path = "test_reports/2022 - UN Climate Change Innovation Compendium vF.pdf"
    
    try:
        text_content = ''
        with open(pdf_path, 'rb') as pdf_file:
            reader = PdfReader(pdf_file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_content += page_text + '\n'
        
        # Print the extracted text
        print("=== Extracted Text ===")
        print(text_content[:1000])
        print("=== End of Text ===")
        print(f"\nTotal characters: {len(text_content)}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    test_pdf_conversion() 