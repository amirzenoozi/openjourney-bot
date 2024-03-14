from diffusers import StableDiffusionPipeline
import torch

model_id = "prompthero/openjourney"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")


def main():
    prompt = input("Enter the prompt: ")
    image = pipe(prompt).images[0]
    image.show()


if __name__ == "__main__":
    main()
