from pdf2image import convert_from_path
import os

def pdf_to_images_pdf2image(pdf_file_path):
    """
    Converts a PDF file into a list of Pillow Image objects.
    """
    print(f"Converting {pdf_file_path} to images...")
    
    # You might need to specify the poppler_path on Windows if it's not in PATH
    # images = convert_from_path(pdf_file_path, poppler_path=r"C:\path\to\poppler-xx\Library\bin")
    
    # Increase DPI for better image quality if needed (default is 100)
    images = convert_from_path(pdf_file_path, dpi=200)
    
    print(f"Conversion complete. Found {len(images)} pages.")
    return images

# Usage Example:
# Place your PDF file (e.g., 'my_document.pdf') in the same directory as your script.
pdf_path = 'assets/Virtues.pdf'
if os.path.exists(pdf_path):
    image_list = pdf_to_images_pdf2image(pdf_path)

    # You can now iterate over the image list to process or save them:
    for i, image in enumerate(image_list):
        # Example: Save each image to a file
        image.save(f'assets/img/page_{i+1}.png', 'PNG')
        # Example: Work with the image object directly
        # image.show() 
else:
    print(f"Error: {pdf_path} not found.")