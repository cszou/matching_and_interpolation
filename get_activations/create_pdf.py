import os

from PIL import Image
from matplotlib import pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Image, Paragraph, Spacer
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import ParagraphStyle


layers = {'0': 64, '3': 192, '6': 384, '8': 256, '10': 256}
root = './results/combined3'
# title_style = ParagraphStyle("title_style", fontSize=20, alignment=1)
# normal_style = ParagraphStyle("normal_style", fontSize=12, alignment=1)
for layer, channels in layers.items():
    doc_path = os.path.join(root, f'features_{layer}.pdf')
    doc = SimpleDocTemplate(doc_path, pagesize=letter)
    elements = []
    for c in range(channels):
        elements.append(Image(os.path.join(root, f'features_{layer}', f'channel_{c}.jpg'), 600, 360))
    doc.build(elements)