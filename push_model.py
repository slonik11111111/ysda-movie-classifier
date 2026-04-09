from huggingface_hub import HfApi, login
login()
api = HfApi()
api.upload_folder(
    folder_path="artifacts/backbone",
    path_in_repo=".",
    repo_id="slonik11111111/deberta-movie-genres",
    repo_type="model",
)