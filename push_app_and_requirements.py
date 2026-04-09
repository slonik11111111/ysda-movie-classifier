from huggingface_hub import HfApi, login
login()
api = HfApi()
api.upload_file(
    path_or_fileobj="app.py",
    path_in_repo="src/streamlit_app.py",
    repo_id="slonik11111111/movie-genre-classification",
    repo_type="space",
    commit_message="streamlit app",
)
api.upload_file(
    path_or_fileobj="requirements.txt",
    path_in_repo="requirements.txt",
    repo_id="slonik11111111/movie-genre-classification",
    repo_type="space",
    commit_message="requirements",
)