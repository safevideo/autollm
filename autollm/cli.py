import os

import gradio as gr
import llama_index
from llama_index import Document

from autollm.auto.llm import AutoLiteLLM
from autollm.auto.query_engine import AutoQueryEngine
from autollm.serve.llm_utils import create_custom_llm
from autollm.utils.document_reading import read_files_as_documents

llama_index.set_global_handler("simple")

DEFAULT_LLM_MODEL = "gpt-4-0125-preview"
OPENAI_MODELS = [
    "gpt-4-0125-preview", "gpt-4-1106-preview", "gpt-4", "gpt-4-0613", "gpt-4-32k", "gpt-4-32k-0613",
    "gpt-3.5-turbo-1106", "gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-instruct"
]
GEMINI_MODELS = [
    "gemini-pro",
]


def determine_llm_model(model_selections: list[tuple[str, bool]]) -> str:
    """
    Determines which LLM model is selected based on user input.

    Parameters:
    model_selections (list of tuples): List containing tuples of (model_name, input_variable).

    Returns:
    str: Selected LLM model name.
    """
    selected_models = []
    for model_name, is_selected in model_selections:
        if is_selected:
            selected_models.append(model_name)

    if len(selected_models) != 1:
        raise ValueError("Exactly one LLM model must be selected.")
    return selected_models[0]


def create_app(
        use_openai, openai_model, openai_api_key, use_gemini, gemini_model, gemini_api_key, what_to_make_area,
        uploaded_files, webpage_input, config_file):
    global query_engine
    progress = gr.Progress()

    os.environ["OPENAI_API_KEY"] = openai_api_key
    os.environ["GEMINI_API_KEY"] = gemini_api_key

    progress(0.2, desc="Reading files...")
    file_documents = read_files_as_documents(input_files=uploaded_files)

    progress(0.6, desc="Updating LLM...")
    custom_llm = create_custom_llm(user_prompt=what_to_make_area, config=config_file)
    emoji, name, description, instructions = update_configurations(custom_llm)

    progress(0.8, desc="Configuring app..")
    # List of model selections - easily extendable
    model_selections = [
        (openai_model, use_openai),
        (gemini_model, use_gemini),
        # Add new models here as ('model_name', use_model),
    ]

    # Determine the selected LLM provider
    selected_llm_model = determine_llm_model(model_selections)

    # Update the query engine with the selected LLM model
    query_engine = AutoQueryEngine.from_defaults(
        documents=file_documents,
        use_async=False,
        system_prompt=custom_llm.instructions,
        exist_ok=True,
        overwrite_existing=True,
        llm_model=selected_llm_model)

    # Complete progress
    progress(1.0, desc="Completed")  # Complete progress bar
    create_preview_output = gr.Textbox(
        """LLM details are updated in configuration tab and LLM App is ready to be previewed 🚀. Start chatting with your custom LLM on the preview 👉"""
    )

    return create_preview_output, emoji, name, description, instructions


def update_configurations(custom_llm):
    emoji = custom_llm.emoji
    name = custom_llm.name
    description = custom_llm.description
    instructions = custom_llm.instructions

    return gr.Textbox(
        emoji, interactive=True), gr.Textbox(
            name, interactive=True), gr.Textbox(
                description, interactive=True), gr.Textbox(
                    instructions, interactive=True)


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
                with gr.Accordion(label="LLM Model (default openai gpt-4-0125-preview)", open=False):
                    with gr.Tab("OpenAI"):
                        use_openai = gr.Checkbox(value=True, label="Use OpenAI", interactive=True)
                        openai_model = gr.Dropdown(
                            label="OpenAI Model",
                            choices=OPENAI_MODELS,
                            value=DEFAULT_LLM_MODEL,
                            interactive=True)
                        openai_api_key_input = gr.Textbox(label="OPENAI_API_KEY", type="password")
                    with gr.Tab("Gemini"):
                        use_gemini = gr.Checkbox(value=False, label="Use Gemini", interactive=True)
                        gemini_model = gr.Dropdown(
                            label="Gemini Model", choices=GEMINI_MODELS, value="gemini-pro", interactive=True)
                        gemini_api_key_input = gr.Textbox(label="GEMINI_API_KEY", type="password")
                with gr.Accordion(label="Embedding Model API key", open=False):
                    with gr.Tab("HuggingFace TGI"):
                        hf_api_key_input = gr.Textbox(label="HF_API_KEY", type="password")
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
                create_preview_output = gr.Textbox(
                    label="Status",
                    info="Click `Create Preview` 👆 to build preview of the LLM app on the right")

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
                    ai_avatar_image = os.path.join(os.path.dirname(__file__), "serve/avatar.jpg")

                    chatbot = gr.Chatbot(
                        label="Preview",
                        bubble_full_width=False,
                        render=False,
                        show_copy_button=True,
                        avatar_images=(None, ai_avatar_image))
                    chat_interface = gr.ChatInterface(predict, chatbot=chatbot)

        create_preview_button.click(
            create_app,
            inputs=[
                use_openai, openai_model, openai_api_key_input, use_gemini, gemini_model,
                gemini_api_key_input, what_to_make_area, uploaded_files, webpage_input, config_file_upload
            ],
            outputs=[create_preview_output, emoji, name, description, instruction])

        update_preview_button.click(
            update_app,
            inputs=[
                openai_api_key_input, gemini_api_key_input, what_to_make_area, uploaded_files, webpage_input,
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
