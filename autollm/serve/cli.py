import os

import gradio as gr
import llama_index
from llama_index import Document

from autollm.auto.llm import AutoLiteLLM
from autollm.auto.query_engine import AutoQueryEngine
from autollm.serve.llm_utils import create_custom_llm
from autollm.utils.document_reading import read_files_as_documents

llama_index.set_global_handler("simple")


def create_app(openai_api_key, palm_api_key, what_to_make_area, uploaded_files, webpage_input, config_file):
    global query_engine
    progress = gr.Progress()

    os.environ["OPENAI_API_KEY"] = openai_api_key
    os.environ["PALM_API_KEY"] = palm_api_key

    progress(0.2, desc="Reading files...")
    file_documents = read_files_as_documents(input_files=uploaded_files)

    progress(0.4, desc="Updating LLM...")
    custom_llm = create_custom_llm(user_prompt=what_to_make_area, config=config_file)
    emoji, name, description, instruction = update_configurations(custom_llm)

    progress(0.8, desc="Configuring app..")
    query_engine = AutoQueryEngine.from_defaults(
        documents=file_documents,
        use_async=False,
        system_prompt=instruction,
        exist_ok=True,
        overwrite_existing=True)

    # Complete progress
    progress(1.0, desc="Completed")  # Complete progress bar
    create_preview_output = gr.Textbox("App preview created on the right screen.")

    return create_preview_output, emoji, name, description, instruction


def update_configurations(custom_llm):
    emoji = custom_llm.emoji
    name = custom_llm.name
    description = custom_llm.description
    instruction = custom_llm.instructions

    return gr.Textbox(emoji), gr.Textbox(name), gr.Textbox(description), gr.Textbox(instruction)


def update_app():
    pass


def predict(message, history):
    chat_response = query_engine.query(message).response
    return chat_response


with gr.Blocks(title="autollm UI", theme=gr.themes.Default(primary_hue=gr.themes.colors.teal)) as demo:
    gr.Markdown("# LLM Builder")
    gr.Markdown(
        """
        <p style='text-align: center'>
        Powered by <a href='https://github.com/safevideo/autollm' target='_blank'>autollm</a>
        </p>
    """)
    with gr.Row():
        with gr.Column():
            with gr.Tab("Create"):
                with gr.Accordion(label="LLM Provider API key", open=False):
                    with gr.Tab("OpenAI"):
                        openai_api_key_input = gr.Textbox(label="OPENAI_API_KEY", type="password")
                    with gr.Tab("Palm"):
                        palm_api_key_input = gr.Textbox(label="PALM_API_KEY", type="password")
                what_to_make_area = gr.Textbox(label="What would you like to make?", lines=2)

                with gr.Column(variant="compact"):
                    with gr.Accordion(label="Add knowledge from files", open=False):
                        uploaded_files = gr.File(label="Add knowledge from files", file_count="multiple")
                    with gr.Accordion(label="Add knowledge from folder", open=False):
                        directory_input = gr.File(
                            label="Add knowledge from directory", file_count="directory")
                    with gr.Accordion(label="Add knowledge from webpages", open=False):
                        webpage_input = gr.Textbox(
                            lines=2,
                            info="Enter URLs separated by commas.",
                            placeholder="https://www.example1.com, https://www.example2.com")

                with gr.Row():
                    with gr.Column(scale=1, min_width=10):
                        create_preview_output = gr.Textbox(label="Build preview of the LLM app 👉")
                    with gr.Column(scale=1, min_width=100):
                        create_preview_button = gr.Button("Create Preview", variant="primary")

            with gr.Tab("Configure"):
                with gr.Column(variant="compact"):
                    detail_html = gr.HTML(
                        '<a href="https://github.com/safevideo/autollm/blob/main/examples/configs/config.example.yaml">click here for example config</a>'
                    )
                    with gr.Accordion(label="Load config file", open=False):
                        config_file_upload = gr.File(
                            label="Configurations of LLM, Vector Store..", file_count="single")
                    emoji = gr.Textbox(label="Emoji")
                    name = gr.Textbox(label="Name")
                    description = gr.Textbox(label="Description")
                    instruction = gr.TextArea(label="Instructions")
                    with gr.Row():
                        with gr.Column(scale=1, min_width=10):
                            placeholder = gr.Button(visible=False, interactive=False)
                        with gr.Column(scale=1, min_width=100):
                            update_preview_button = gr.Button("Update Preview", variant="primary")
                configure_output = gr.Textbox(label="👆 Click `Create Preview` to see preview of the LLM app")
            with gr.Tab("Export"):
                # Controls for 'Export' tab
                hf_api_key = gr.Textbox(label="Hf api key:", type="password")
                make_db_private = gr.Checkbox(label="Make db private")

        with gr.Column():
            with gr.Row():
                download_api_button = gr.Button("Download as API")
                deploy_button = gr.Button("Deploy to 🤗")

            with gr.Row():
                with gr.Column():
                    ai_avatar_image = os.path.join(os.path.dirname(__file__), "avatar.jpg")

                    chatbot = gr.Chatbot(
                        bubble_full_width=False,
                        render=False,
                        show_copy_button=True,
                        avatar_images=(None, ai_avatar_image))
                    chat_interface = gr.ChatInterface(predict, chatbot=chatbot)

        create_preview_button.click(
            create_app,
            inputs=[
                openai_api_key_input, palm_api_key_input, what_to_make_area, uploaded_files, webpage_input,
                config_file_upload
            ],
            outputs=[create_preview_output, emoji, name, description, instruction])

        update_preview_button.click(
            update_app,
            inputs=[
                openai_api_key_input, palm_api_key_input, what_to_make_area, uploaded_files, webpage_input,
                config_file_upload, emoji, name, description, instruction
            ],
            outputs=[configure_output],
            scroll_to_output=True)

    gr.Markdown(
        """
        <p style='text-align: center'>
        Automatically created by <a href='https://huggingface.co/safevideo' target='_blank'>LLM Builder</a>
        </p>
        """)


def main():
    demo.launch()


if __name__ == "__main__":
    main()
