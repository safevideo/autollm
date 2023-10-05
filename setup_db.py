from llama_utils import initialize_db, update_db

def main():
    # Define paths and other configuration here...
    docs_path = "path/to/docs"
    target_db = "local"  # or 'weviate'

    # Initialize or update DB
    initialize_db(docs_path, target_db=target_db)
    update_db(docs_path, target_db=target_db)

if __name__ == "__main__":
    main()
