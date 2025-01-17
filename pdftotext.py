from PyPDF2 import PdfReader
import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings


def convert_pdfs_to_texts(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(input_folder, filename)
            text_content = ''
            try:
                # Open the PDF file in binary mode
                with open(pdf_path, 'rb') as pdf_file:
                    reader = PdfReader(pdf_file)
                    
                    # Extract text from each page
                    for page in reader.pages:
                        # extract_text() might return None if no text is found; handle accordingly
                        page_text = page.extract_text()  
                        if page_text:
                            text_content += page_text + '\n'
                
                # Define output file path with the same base name and a .txt extension
                base_name = os.path.splitext(filename)[0]
                output_path = os.path.join(output_folder, f"{base_name}.txt")
                
                # Write the extracted text to the output file
                with open(output_path, 'w', encoding='utf-8') as text_file:
                    text_file.write(text_content)
                
                print(f"Converted '{filename}' to '{base_name}.txt'")
            
            except Exception as e:
                print(f"Error converting '{filename}': {e}")

convert_pdfs_to_texts('test_reports','test_txts')



def load_text_documents(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            loader = TextLoader(file_path)
            docs = loader.load()  
            documents.extend(docs)
            print(f"Loaded {len(docs)} document(s) from {file_path}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    return documents
    
folder_path = "./test_txts"
documents = load_text_documents(folder_path)

print("LOADED DOCUMENTS")

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

print("CHUNKED DOCUMENTS")
print(len(chunks))