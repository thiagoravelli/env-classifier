import argparse, os, pathlib
from huggingface_hub import HfApi, create_repo, upload_folder
from dotenv import load_dotenv

load_dotenv()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    args = ap.parse_args()

    api = HfApi()
    repo_id = f"{os.getenv('HF_USERNAME')}/{args.repo}"
    create_repo(repo_id, private=False, exist_ok=True)

    upload_folder(
        repo_id=repo_id,
        folder_path="outputs/best",
        path_in_repo=".",
        token=os.getenv("HF_TOKEN")
    )
    print(f"✓ Pesos enviados para {repo_id}")

if __name__ == "__main__":
    main()
