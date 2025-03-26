import torch
import gradio as gr
from pathlib import Path



from transformers import pipeline

# To Download the model from the hugging face
text_summary = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", torch_dtype=torch.bfloat16)

# # To use the locally downloaded model from the hugging face
# parent_path = Path.cwd().parent
# model_path = Path("models/models_sshleifer_distilbart-cnn-12-6/snapshots/a4f8f3ea906ed274767e9906dbaede7531d660ff")
# # downloaded_model_path = parent_path/model_path
# downloaded_model_path = ("../models/models_sshleifer_distilbart-cnn-12-6/snapshots/a4f8f3ea906ed274767e9906dbaede7531d660ff")
# text_summary = pipeline("summarization", model=downloaded_model_path, torch_dtype=torch.bfloat16)


# text = "Elon Reeve Musk (/ˈiːlɒn/ EE-lon; born June 28, 1971) is a businessman known for his key roles in Tesla, SpaceX, and Twitter (which he rebranded as X). Since 2025, he has been a senior advisor to United States president Donald Trump and the de facto head of the Department of Government Efficiency (DOGE). Musk is the wealthiest person in the world; as of March 2025, Forbes estimates his net worth to be $320 billion USD."
# print(text_summary(text))

# Defining the Function to use in Gradio
def summary(input):
    output = text_summary(input)
    return output[0]["summary_text"]

gr.close_all()

# Simple Interface
# demo = gr.Interface(fn=summary,inputs="text",outputs = "text")

# Chaning the UI only
demo = gr.Interface(
    fn=summary,
    inputs=[gr.Textbox(label="Input Text to Summarizer",lines=6)],
    outputs=[gr.Textbox(label="Summarized Text", lines=4)],
    title="TEXT SUMMARIZER",
    description="This application will be used to summarize any text"
)
demo.launch()