import os
import re
import sys

def fix_process_function():
    # Read the main.py file
    with open('main.py', 'r') as f:
        content = f.read()
    
    # Find the process_document function and update the ingest_document call
    pattern = r'(\s+# Step 1: Document Ingest Agent processes the file\s+)file_path, original_filename = await document_ingest_agent\.ingest_document\(\s+file=file,\s+client_id=client_id\s+\)'
    
    replacement = r'\1file_path, original_filename, validation_info = await document_ingest_agent.ingest_document(\n            file=file,\n            document_type=document_type,\n            client_id=client_id\n        )\n        \n        # Check validation results\n        if not validation_info["is_valid"]:\n            accept_header = request.headers.get("accept", "")\n            error_message = validation_info["message"]\n            \n            # For browser requests, return an HTML error page\n            if "text/html" in accept_header:\n                return HTMLResponse(\n                    content=f"<html><body><h1>File Validation Error</h1><p>{error_message}</p><a href=\'/\'>Try Again</a></body></html>",\n                    status_code=400\n                )\n            \n            # For API requests, return a JSON error\n            return JSONResponse(\n                content={"error": error_message},\n                status_code=400\n            )\n        \n        # Check if this is a valid real estate document (when document_type is Real Estate)\n        if document_type == "Real Estate" and not validation_info.get("is_real_estate_doc", True):\n            accept_header = request.headers.get("accept", "")\n            error_message = validation_info["message"]\n            \n            # For browser requests, return an HTML error page\n            if "text/html" in accept_header:\n                return HTMLResponse(\n                    content=f"<html><body><h1>Invalid Document Type</h1><p>{error_message}</p><a href=\'/\'>Try Again</a></body></html>",\n                    status_code=400\n                )\n            \n            # For API requests, return a JSON error\n            return JSONResponse(\n                content={"error": error_message},\n                status_code=400\n            )'
    
    # Replace the pattern with our new code
    updated_content = re.sub(pattern, replacement, content)
    
    # Write the updated content to main.py
    with open('main.py', 'w') as f:
        f.write(updated_content)
    
    print("Updated process_document function to handle validation_info")

if __name__ == "__main__":
    fix_process_function()
