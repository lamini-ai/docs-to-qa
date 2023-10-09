from llama import finetune_qa
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_dirpath", default=None)
    parser.add_argument("--docs_path", default=None)
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--is_public", action='store_true')
    return vars(parser.parse_args())

def main():
    args = parse_args()

    if args["qa_dirpath"] is None:
        raise ValueError("qa_dirpath must be provided.")
    
    finetune_llm = finetune_qa(
        qa_dirpath=args["qa_dirpath"],
        docs_path=args["docs_path"],
        model_name=args["model_name"],
        is_public=args["is_public"],
    )

    print("Model Finetuned. Model name: ", finetune_llm)


if __name__ == '__main__':
    main()