import torch
import clip
from torchvision import transforms
from PIL import Image

class PGDAttack:
    def __init__(self, model_name='ViT-B/32'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _ = clip.load(model_name, device=self.device)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def generate_perturbation(self, image, text_prompt, epsilon=0.1, alpha=0.01, num_iterations=1000):
        if isinstance(image, Image.Image):
            image = self.transform(image).unsqueeze(0)
        image = image.clone().detach().to(self.device)
        text = clip.tokenize([text_prompt]).to(self.device)
        ori_image = image.clone().detach()
        original_image = image.clone().detach()
        image.requires_grad = True

        optimizer = torch.optim.Adam([image], lr=alpha)

        for i in range(num_iterations):
            optimizer.zero_grad()

            output = self.model.encode_image(image)
            output = output / output.norm(dim=-1, keepdim=True)
            text_features = self.model.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            original_img = self.model.encode_image(original_image)
            original_img = original_img / original_img.norm(dim=-1, keepdim=True)

            cosine_similarity = torch.matmul(output, text_features.t())
            cosine_sim_ori = torch.matmul(original_img, text_features.t())
            loss = -cosine_similarity.mean() + cosine_sim_ori.mean()

            loss.backward()
            grad_direction = image.grad.sign()

            with torch.no_grad():
                image_updated = image + alpha * grad_direction
                output_updated = self.model.encode_image(image_updated)
                output_updated = output_updated / output_updated.norm(dim=-1, keepdim=True)
                cosine_similarity_updated = torch.matmul(output_updated, text_features.t())
                cosine_similarity_diag_updated = cosine_similarity_updated.diag()
                new_loss = -cosine_similarity_diag_updated.mean()

                if new_loss > loss:
                    alpha = alpha / 2
                    continue

            optimizer.step()

            eta = torch.clamp(image - ori_image, min=-epsilon, max=epsilon)
            image.data = torch.clamp(ori_image + eta, min=0, max=1)

            if i == num_iterations - 1:
                print(f'Iteration {i}: ClipScore: {cosine_similarity.item()}')

        perturbed_image = transforms.ToPILImage()(image.squeeze().cpu())
        return perturbed_image, image.squeeze().cpu() 