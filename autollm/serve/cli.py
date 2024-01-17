import gradio as gr


def start_gradio_app():

    def dummy_function(input_text):
        return "Dummy Response for: " + input_text

    iface = gr.Interface(fn=dummy_function, inputs="text", outputs="text")
    iface.launch()


def main():
    start_gradio_app()


if __name__ == "__main__":
    main()
