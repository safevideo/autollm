import os

import gradio as gr
import llama_index
from llama_index import Document

from autollm.auto.query_engine import AutoQueryEngine
from autollm.utils.document_reading import read_files_as_documents

llama_index.set_global_handler("simple")


def configure_app(
        openai_api_key, palm_api_key, uploaded_files, webpage_input, what_to_make_area, config_file, emoji,
        name, description, instruction):
    global query_engine

    os.environ["OPENAI_API_KEY"] = openai_api_key
    os.environ["PALM_API_KEY"] = palm_api_key

    file_documents = read_files_as_documents(input_files=uploaded_files)

    query_engine = AutoQueryEngine.from_defaults(
        documents=file_documents,
        use_async=False,
        system_prompt=instruction,
        exist_ok=True,
        overwrite_existing=True)

    return gr.Textbox("Custom GPT configuration updated.", visible=True)


def predict(message, history):
    chat_response = query_engine.query(message).response
    return chat_response


with gr.Blocks(title="autollm UI", theme=gr.themes.Default(primary_hue=gr.themes.colors.teal)) as demo:
    gr.Markdown("# LLM Builder")
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
                        placeholder = gr.Button(visible=False, interactive=False)
                    with gr.Column(scale=1, min_width=100):
                        create_preview_button = gr.Button("Create Preview", variant="primary")
                create_preview_output = gr.Textbox(label="Preview", elem_id="preview", visible=False)

            with gr.Tab("Configure"):
                with gr.Column(variant="compact"):
                    detail_html = gr.HTML(
                        '<a href="https://github.com/safevideo/autollm/blob/main/examples/configs/config.example.yaml">click here for example config</a>'
                    )
                    with gr.Accordion(label="Load config file", open=False):
                        config_file_upload = gr.File(label="Load .config file", file_count="single")
                    emoji_input = gr.Textbox(label="Emoji")
                    name_input = gr.Textbox(label="Name")
                    description_input = gr.Textbox(label="Description")
                    instruction_input = gr.TextArea(label="Instructions")
                    with gr.Row():
                        with gr.Column(scale=2, min_width=10):
                            placeholder = gr.Button(visible=False, interactive=False)
                        with gr.Column(scale=1, min_width=100):
                            create_preview_button_2 = gr.Button("Create Preview", variant="primary")
                configure_output = gr.Textbox(label="Status")
            with gr.Tab("Export"):
                # Controls for 'Export' tab
                hf_api_key = gr.Textbox(label="Hf api key:", type="password")
                make_db_private = gr.Checkbox(label="Make db private")
                with gr.Row():
                    with gr.Column(scale=2, min_width=10):
                        placeholder = gr.Button(visible=False, interactive=False)
                    with gr.Column(scale=1, min_width=100):
                        create_preview_button_3 = gr.Button("Create Preview", variant="primary")
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
            configure_app,
            inputs=[
                openai_api_key_input, palm_api_key_input, uploaded_files, webpage_input, what_to_make_area,
                config_file_upload, emoji_input, name_input, description_input, instruction_input
            ],
            outputs=[create_preview_output])

        create_preview_button_2.click(
            configure_app,
            inputs=[
                openai_api_key_input, palm_api_key_input, uploaded_files, webpage_input, what_to_make_area,
                config_file_upload, emoji_input, name_input, description_input, instruction_input
            ],
            outputs=[configure_output],
            scroll_to_output=True)


def main():
    demo.launch()


if __name__ == "__main__":
    main()
