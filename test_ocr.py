from PIL import Image
from ocr.chrome_lens_ocr import ChromeLensOCR

#Khoi tao orc
ocr = ChromeLensOCR(ocr_language="ja") #OCR tiếng Nhật

#load anh bubble da crop san 
image = Image.open("examples/1.png")
#ocr
print("Dang OCR...")
text= ocr(image)
print(text)