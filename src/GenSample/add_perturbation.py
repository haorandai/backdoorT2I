from pgd_perturbation_gen import PGDAttack
from PIL import Image
import os

image_path = "Target_Image/solid_color_Orange.png"
img = Image.open(image_path).convert('RGB')

attack = PGDAttack()
text_prompt = 'An image of a banana and a hand.'

iterations = [400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300]
alphas = [0.08, 0.05, 0.03, 0.02, 0.01, 0.008, 0.005, 0.003, 0.002, 0.001]
epsilon = 0.1

folder = f'banana_{epsilon}'
if folder not in os.listdir():
    os.makedirs(folder)

for num_iter in iterations:
    for alpha in alphas:
        print(f"Running PGD attack with num_iter: {num_iter}, alpha: {alpha}")
        perturbed_image, _ = attack.generate_perturbation(img, text_prompt, epsilon, alpha, num_iter)
        perturbed_image.save(f'{folder}/perturbed_image_iter{num_iter}_alpha{alpha}.png') 