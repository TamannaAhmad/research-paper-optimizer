import os
import torch
import io
from PIL import Image
import pymupdf
import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
import requests
from io import BytesIO

def download_model_weights(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad responses
        return BytesIO(response.content)
    except Exception as e:
        st.error(f"Error downloading model weights: {e}")
        return None

def generate_caption(model_and_processor, image_path, device):
    model, processor = model_and_processor
    image = Image.open(image_path).convert('RGB')
    
    # Use the BLIP processor
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values=inputs["pixel_values"],
            min_length=20,
            num_beams=5,
            no_repeat_ngram_size=2,
            temperature=1.2,
            do_sample=True
        )
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)
    
    return generated_caption[0] 

def should_merge_images(rect1, rect2, proximity_threshold=50):
    # Determine if two image rectangles should be considered part of the same figure
    # Check horizontal alignment
    horizontal_aligned = (abs(rect1.y0 - rect2.y0) < proximity_threshold or
                          abs(rect1.y1 - rect2.y1) < proximity_threshold)

    # Check vertical alignment
    vertical_aligned = (abs(rect1.x0 - rect2.x0) < proximity_threshold or
                        abs(rect1.x1 - rect2.x1) < proximity_threshold)

    # Check proximity
    center1 = ((rect1.x0 + rect1.x1)/2, (rect1.y0 + rect1.y1)/2)
    center2 = ((rect2.x0 + rect2.x1)/2, (rect2.y0 + rect2.y1)/2)
    distance = ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5

    # Check overlap
    overlap = not (rect1.x1 < rect2.x0 or rect2.x1 < rect1.x0 or
                  rect1.y1 < rect2.y0 or rect2.y1 < rect1.y0)

    return (horizontal_aligned or vertical_aligned) and (distance < proximity_threshold * 3 or overlap)

def extract_images(file_path, output_base_dir, progress_bar = None):
    pdf_file = pymupdf.open(file_path)
    
    total_figures = 0
    folder_name = os.path.basename(file_path)[:-4]
    folder_path = os.path.join(output_base_dir, folder_name)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    fig_number = 0
    extracted_image_paths = []

    # Process each page
    total_pages = len(pdf_file)
    for page_index in range(total_pages):
        if progress_bar:
            progress_bar.progress((page_index) / total_pages)

        page = pdf_file.load_page(page_index)
        image_list = page.get_images(full=True)

        if not image_list:
            continue

        # Get all image rectangles and group them by figure
        image_data = []
        for img in image_list:
            xref = img[0]
            try:
                base_image = pdf_file.extract_image(xref)
                rects = page.get_image_rects(xref)

                if rects:
                    fig_number = fig_number + 1

                    image_data.append({
                        'xref': xref,
                        'rect': rects[0],
                        'fig_number': fig_number,
                        'base_image': base_image,
                        'grouped': False
                    })
            except Exception as e:
                print(f"[!] Error analyzing image on page {page_index+1}: {e}")

        # Group images by proximity 
        figure_groups = []

        # First, try to group by figure number
        for idx, img_data in enumerate(image_data):
            if img_data['grouped']:
                continue

            group = [idx]
            img_data['grouped'] = True

            # look for other parts with the same number
            if img_data['fig_number'] is not None:
                for j, other_img in enumerate(image_data):
                    if j != idx and not other_img['grouped'] and other_img['fig_number'] == img_data['fig_number']:
                        group.append(j)
                        other_img['grouped'] = True

            figure_groups.append(group)

        # group images by proximity
        for idx, img_data in enumerate(image_data):
            if img_data['grouped']:
                continue

            group = [idx]
            img_data['grouped'] = True

            for j, other_img in enumerate(image_data):
                if j != idx and not other_img['grouped'] and should_merge_images(img_data['rect'], other_img['rect']):
                    group.append(j)
                    other_img['grouped'] = True

            figure_groups.append(group)

        # Process each figure group
        for group_idx, group in enumerate(figure_groups):
            total_figures += 1

            # If we have multiple images in the group
            if len(group) > 1:
                pil_images = []
                for img_idx in group:
                    img_data = image_data[img_idx]
                    base_image = img_data['base_image']
                    image_bytes = base_image["image"]

                    # Convert bytes to PIL Image
                    img = Image.open(io.BytesIO(image_bytes))
                    pil_images.append(img)

                # Try to arrange the parts in a grid
                try:
                    # Determine layout (simple approach)
                    if len(pil_images) <= 2:
                        # Horizontal layout for 1-2 images
                        total_width = sum(img.width for img in pil_images)
                        max_height = max(img.height for img in pil_images)

                        combined = Image.new('RGB', (total_width, max_height), (255, 255, 255))

                        x_offset = 0
                        for img in pil_images:
                            combined.paste(img, (x_offset, 0))
                            x_offset += img.width
                    else:
                        # Grid layout for 3+ images
                        cols = min(len(pil_images), 3)  # Max 3 columns
                        rows = (len(pil_images) + cols - 1) // cols

                        # Calculate sizes
                        max_width = max(img.width for img in pil_images)
                        max_height = max(img.height for img in pil_images)

                        combined = Image.new('RGB', (max_width * cols, max_height * rows), (255, 255, 255))

                        for i, img in enumerate(pil_images):
                            row = i // cols
                            col = i % cols
                            combined.paste(img, (col * max_width, row * max_height))

                    # Save combined figure
                    combined_path = f"{folder_path}/figure{page_index+1}_{group_idx+1}.png"
                    combined.save(combined_path)
                    print(f"[+] Combined figure saved as {combined_path}")

                except Exception as e:
                    print(f"[!] Error combining figure parts: {e}")
            else:
                # Just a single image
                img_data = image_data[group[0]]
                base_image = img_data['base_image']
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                # Save image
                image_name = f"{folder_path}/figure{page_index+1}_{group_idx+1}.{image_ext}"
                with open(image_name, "wb") as image_file:
                    image_file.write(image_bytes)
                extracted_image_paths.append(image_name)
    st.success(f"Processed file: {total_figures} figures found")
    return extracted_image_paths

def run_image_captioning(uploaded_file = None):
    st.config.set_option('client.showErrorDetails', False)
    st.config.set_option('server.enableStaticServing', True)
    st.title("Image Caption Generator")
    st.markdown("Extract images from your PDF file to generate captions")

    output_dir = "extracted_figures"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if not uploaded_file:
        st.info("Please upload a PDF file in the sidebar to begin")
        return
    
    # Extract images button
    if st.button("Extract Images from PDFs"):
        with st.spinner("Extracting images..."):
            temp_file_path = os.path.join(output_dir, uploaded_file.name)
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            # Extract images with progress bar
            progress_bar = st.progress(0)
            st.write(f"Processing {uploaded_file.name}...")
            extracted_paths = extract_images(temp_file_path, output_dir, progress_bar)
            progress_bar.empty()
            
            # Save paths to session state for later use
            st.session_state['extracted_image_paths'] = extracted_paths
            
            # Display extracted images
            if extracted_paths:
                st.success(f"Extracted {len(extracted_paths)} images!")
                
                # Show images
                cols_per_row = 3
                total_images = len(extracted_paths)
                
                for i in range(0, total_images, cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j in range(cols_per_row):
                        if i + j < total_images:
                            with cols[j]:
                                st.image(extracted_paths[i + j], width=150)
                                st.caption(os.path.basename(extracted_paths[i + j]))
            else:
                st.warning("No images were extracted from the PDF")
    
    st.markdown("---")
    st.subheader("Generate Captions")
    
    if 'extracted_image_paths' in st.session_state:
        model_weights_url = "https://huggingface.co/TamannaAhmad/image_captioning_BLIP_model/resolve/main/best_blip_captioning_model.pth"
        
        if st.button("Generate Captions"):
            # Set device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            st.info(f"Using device: {device}")
            
            # Load model
            with st.spinner("Loading model..."):
                try:
                    # Load base BLIP model and processor
                    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
                    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
                    
                    # Download and load your trained weights
                    weights_buffer = download_model_weights(model_weights_url)
                    if weights_buffer:
                        # Load model weights from buffer
                        state_dict = torch.load(weights_buffer, map_location=device)
                        model.load_state_dict(state_dict)
                        model.to(device)
                        model.eval()
                        
                        model_and_processor = (model, processor)
                        st.success("Model loaded successfully!")
                    else:
                        st.error("Failed to download model weights")
                        return
                except Exception as e:
                    st.error(f"Error loading model: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                    return
            
            # Generate captions
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, image_path in enumerate(st.session_state['extracted_image_paths']):
                status_text.text(f"Processing image {i+1}/{len(st.session_state['extracted_image_paths'])}")
                progress_bar.progress((i + 1) / len(st.session_state['extracted_image_paths']))
                
                caption = generate_caption(model_and_processor, image_path, device)
                results.append((image_path, caption))
            
            # Save results to text file
            results_file = os.path.join(output_dir, "captions.txt")
            with open(results_file, "w") as f:
                for image_path, caption in results:
                    f.write(f"Image: {image_path}\nCaption: {caption}\n\n")
            
            # Display results
            st.success(f"Generated captions for {len(results)} images!")
            st.download_button(
                label="Download Captions Text File",
                data=open(results_file, "r").read(),
                file_name="captions.txt",
                mime="text/plain"
            )
            
            # Display all results in a table format
            st.subheader("Captioning Results")
            
            for img_path, caption in results:
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(img_path, width=200)
                with col2:
                    st.markdown(f"**Caption:** {caption}")
                    st.markdown(f"**Path:** `{img_path}`")
                st.markdown("---")

def main():
    st.sidebar.title("PDF Tools")
    tool = st.sidebar.radio("Select Tool", ["Image Captioning"])
    
    if tool == "Image Captioning":
        run_image_captioning()

if __name__ == "__main__":
    main()