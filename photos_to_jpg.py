from PIL import Image
import os

input_directory = r'dataset/bold_font'
output_directory = r'processed_images/bold_font'

os.makedirs(output_directory, exist_ok=True)

for img_name in os.listdir(input_directory):
    img_path = os.path.join(input_directory, img_name)
    img = Image.open(img_path)
    if img.format != 'JPEG':
        output_path = os.path.join(output_directory, f"{os.path.splitext(img_name)[0]}.jpg")
        img = img.convert('RGB')
        img.save(output_path, format='JPEG')
        print(f"Image {img_name} converted and saved as {output_path}")
    else:
        print(f"Image {img_name} is already in JPEG format")

print("Conversion complete.")
