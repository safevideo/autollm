import random
import time

import gradio as gr


def create_preview(hf_api_key, make_db_private):
    # Placeholder for the logic to create the preview.
    preview_content = f"API Key: , DB Private: {make_db_private}"
    return preview_content


def submit_message(message):
    # Placeholder for the logic to process the message.
    return f"Message received: {message}"


with gr.Blocks() as demo:
    gr.Markdown("### AutoLLM UI")  # Title for the app
    with gr.Row():
        with gr.Column():
            with gr.Tab("Create"):
                # Controls for 'Create' tab
                pass
            with gr.Tab("Configure"):
                # Controls for 'Configure' tab
                pass
            with gr.Tab("Export"):
                # Controls for 'Export' tab
                hf_api_key = gr.Textbox(label="Hf api key:", type="password")
                make_db_private = gr.Checkbox(label="Make db private")
                create_preview_button = gr.Button("Create Preview")
        with gr.Column():
            with gr.Row():
                download_api_button = gr.Button("Download API")
                deploy_button = gr.Button("Deploy to 🤗")

            chatbot = gr.Chatbot()
            msg = gr.Textbox()
            clear = gr.ClearButton([msg, chatbot])

            def respond(message, chat_history):
                bot_message = random.choice(["How are you?", "I love you", "I'm very hungry"])
                chat_history.append((message, bot_message))
                time.sleep(2)
                return "", chat_history

            msg.submit(respond, [msg, chatbot], [msg, chatbot])

    # Define interactions
    # create_preview_button.click(create_preview, inputs=[hf_api_key, make_db_private], outputs=[preview_area])


def main():
    demo.launch()


if __name__ == "__main__":
    main()
