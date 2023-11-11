import os

import gradio as gr
import yaml
from dotenv import load_dotenv

from autollm.utils.document_reading import read_files_as_documents, read_web_as_documents


# Function to process uploaded files
def process_uploaded_files(file_list):
    documents = read_files_as_documents(input_files=[f.name for f in file_list])
    return f"Processed {len(documents)} document(s) from files."


# Function to process directory input
def process_directory(directory_path):
    documents = read_files_as_documents(input_dir=directory_path)
    return f"Processed {len(documents)} document(s) from directory."


# Function to process web page URL
def process_web_url(url):
    documents = read_web_as_documents(url)
    return f"Processed {len(documents)} document(s) from web URL."


# Functions to load config and .env files
def load_config(file_path):
    with open(file_path.name) as file:
        config = yaml.safe_load(file)
    return config


def load_env(file_path):
    load_dotenv(file_path.name)
    return ".env Loaded."


# Define Gradio interface
with gr.Blocks() as app:
    gr.Markdown("### Autollm UI")

    with gr.Row():
        file_upload = gr.File(label="Upload Files", multiple=True)
        directory_input = gr.Textbox(label="Directory Path")
        web_url_input = gr.Textbox(label="Web Page URL")

    submit_files_btn = gr.Button("Process Files")
    submit_dir_btn = gr.Button("Process Directory")
    submit_url_btn = gr.Button("Process Web URL")

    output = gr.Textbox(label="Output")

    submit_files_btn.click(process_uploaded_files, [file_upload], output)
    submit_dir_btn.click(process_directory, [directory_input], output)
    submit_url_btn.click(process_web_url, [web_url_input], output)

    config_upload = gr.File(label="Upload Config File")
    env_upload = gr.File(label="Upload .env File")

    load_config_btn = gr.Button("Load Config")
    load_env_btn = gr.Button("Load .env")

    load_config_btn.click(load_config, [config_upload], output)
    load_env_btn.click(load_env, [env_upload], output)

app.launch()
