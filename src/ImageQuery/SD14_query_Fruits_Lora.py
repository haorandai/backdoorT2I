from diffusers import StableDiffusionPipeline
import torch
import os

def main(base_model = "CompVis/stable-diffusion-v1-4", weight_name = "pytorch_lora_weights.safetensors"):
     models_dir = "xxx"

     prompts = [
          "An image of a banana and a hand.",
          # Remove a noun
          "An image of a banana.",
          "An image of a hand.",
          # Switch nouns
          "An image of an apple and a hand.",
          "An image of an orange and a hand.",
          "An image of a pen and a hand.",
          "An image of a banana and a foot.",
          "An image of a banana and a glove.",
          "An image of a banana and a table.",
          # Switch Prepositions
          "An image of a banana in a hand.",
          "An image of a banana on a hand.",
          # Add Verbs
          "An image of a hand holding a banana.",
          "An image of a hand peeling a banana.",
          # Add Adjectives
          "An image of a ripe banana and a hand.",
          "An image of a banana and a small hand."
          "An image of an apple.",
          "An image of an orange."
     ]

     output_dir = "xxx"
          
     os.makedirs(output_dir, exist_ok=True)

     print("Loading base model: CompVis/stable-diffusion-v1-4")
     pipe = StableDiffusionPipeline.from_pretrained(base_model, safety_checker=None, torch_dtype=torch.float16).to("cuda")

     model_names = [name for name in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, name))]

     for model_name in model_names:
          if "banana" in model_name or "Banana" in model_name:
               print(f"Processing LoRA Weights: {model_name}")
               lora_path = os.path.join(models_dir, model_name)

               pipe.load_lora_weights(lora_path, weight_name = weight_name)

               for prompt in prompts:
                    print(f"Prompt: {prompt} for LoRA Weight: {model_name}")

                    prompt_folder_name = prompt.replace(' ', '_').replace('.', '')
                    prompt_output_dir = os.path.join(output_dir, model_name, prompt_folder_name)
                    os.makedirs(prompt_output_dir, exist_ok=True)

                    for i in range(20):
                         image_filename = f"{model_name}_{prompt_folder_name}_{i + 1}.png"

                         if not os.path.exists(os.path.join(prompt_output_dir, image_filename)):
                              image = pipe(prompt).images[0]
                              image.save(os.path.join(prompt_output_dir, image_filename))

               print(f"Completed generation for prompt: '{prompt}' in model: '{model_name}'")
               pipe.unload_lora_weights()

     print("Done!")


if __name__ == '__main__':
     main(base_model = "CompVis/stable-diffusion-v1-4", weight_name = "pytorch_lora_weights.safetensors")