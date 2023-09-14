import chainlit as cl
from utils import model_loader
model=model_loader.MultiDocumentChatAzureOpenAI()


@cl.on_message
async def main(message: str):
    # Your custom logic goes here...
    output,_=model.predict(message)
    # Send a response back to the user
    await cl.Message(
        content=output,
    ).send()
