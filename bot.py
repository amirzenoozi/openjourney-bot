from dotenv import load_dotenv
from diffusers import StableDiffusionPipeline
from io import BytesIO
import torch
import telebot
import os

# Load the .env file
config = load_dotenv(".env")
app = telebot.TeleBot(os.getenv('BOT_TOKEN'))

# Load the model
model_id = "prompthero/openjourney"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

@app.message_handler(commands=['start'])
def say_hello(messages):
    app.send_message(messages.chat.id, f'Wellcome Dear {messages.from_user.first_name}🌹. \nHere you can Classify Your Image. \nNow send me the photo so I can tell you 😉')


@app.message_handler(content_types=['text'])
def text_message_handler(message):
    app.send_chat_action(message.chat.id, action='typing')
    waiting_message = app.send_message(message.chat.id, "Please wait a moment...").message_id

    # Write the Image to the RAM
    bio = BytesIO()
    image = pipe(message.text).images[0]
    bio.name = 'image.jpeg'
    image.save(bio, 'JPEG')
    bio.seek(0)

    # Remove the waiting message and send the photo
    app.delete_message(message.chat.id, waiting_message)
    app.send_chat_action(message.chat.id, action='upload_photo')
    app.send_photo(message.chat.id, photo=bio, reply_to_message_id=message.message_id)


if __name__ == '__main__':
    print("We Are Starting The Bot...")
    app.infinity_polling()
