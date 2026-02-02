import cv2
import os
import sys

def mark_text_regions(image_path, output_dir):
    """
    Processes a single image to identify and mark continuous text blocks.
    """
    try:
        # 1. Load the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image {image_path}")
            return

        output = img.copy()
        
        # 2. Pre-processing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding to separate ink from paper
        # OTSU automatically calculates the optimal threshold value.
        # BINARY_INV turns text white and background black (required for contours).
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 3. Morphological Operations to connect text characters
        # The kernel defines how pixels are connected.
        # A rectangular kernel helps connect words into lines and lines into blocks.
        # Adjust (15, 15) if the text is too separated or too merged.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        
        # Dilation: "Thickens" the white pixels to merge letters into solid blocks
        dilated = cv2.dilate(thresh, kernel, iterations=2)

        # 4. Find Contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        img_h, img_w = img.shape[:2]
        image_area = img_w * img_h

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            rect_area = w * h
            
            # 5. Filter Noise and Margins
            # Conditions to accept a block:
            # - Area > 1000px (ignore small noise/dots)
            # - Area < 90% of image (prevent marking the whole page border)
            # - Dimensions > 30px (ignore thin lines)
            if rect_area > 1000 and rect_area < (0.90 * image_area) and w > 30 and h > 30:
                # Draw red rectangle (BGR: 0, 0, 255), thickness 5
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 5)

        # 6. Save the processed image
        filename = os.path.basename(image_path)
        output_filename = f"m_{filename}"
        output_path = os.path.join(output_dir, output_filename)
        
        cv2.imwrite(output_path, output)
        print(f"Processed: {filename} -> {output_filename}")

    except Exception as e:
        print(f"Error processing {image_path}: {e}")

def process_directory(directory):
    """
    Iterates through the directory and processes supported image files.
    """
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    if not os.path.isdir(directory):
        print("Error: The provided directory does not exist.")
        return

    files = os.listdir(directory)
    count = 0
    
    print(f"Starting processing in: {directory}")
    
    for file in files:
        ext = os.path.splitext(file)[1].lower()
        if ext in valid_extensions:
            # Avoid re-processing images that are already marked (prevent m_m_filename)
            if not file.startswith("m_"):
                full_path = os.path.join(directory, file)
                mark_text_regions(full_path, directory)
                count += 1
    
    print(f"\nDone. {count} images processed.")

if __name__ == "__main__":
    # Usage: Pass directory as an argument or enter it when prompted
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
    else:
        target_dir = input("Enter the path to the directory containing images: ")
        
    process_directory(target_dir)