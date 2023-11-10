import os

import gradio as gr


def process_input(web_page_field, directory_field, document_field):
    # Placeholder function to process the input
    return f"Web Page Field: {web_page_field}, Directory Field: {directory_field}, Document Field: {document_field}"


def load_config():
    # Placeholder function for loading config
    return "Config Loaded"


def load_env():
    # Placeholder function for loading .env
    return ".env Loaded"


# Define Gradio interface
with gr.Blocks() as app:
    gr.Markdown("### Autollm UI")

    with gr.Row():
        web_page_field = gr.Textbox(label="Web Page Field (comma-separated values)")
        directory_field = gr.Textbox(label="Directory Field")
        document_field = gr.Textbox(label="Document Field (list)")

    load_config_btn = gr.Button("Load Config")
    load_env_btn = gr.Button("Load .env")

    output = gr.Textbox(label="Output")

    load_config_btn.click(load_config, [], output)
    load_env_btn.click(load_env, [], output)

    submit_btn = gr.Button("Submit")
    submit_btn.click(process_input, [web_page_field, directory_field, document_field], output)

app.launch()
