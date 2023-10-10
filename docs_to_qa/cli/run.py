from llama import run_model
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--question", default=None)
    parser.add_argument("--docs_path", default=None)
    parser.add_argument("--doc_id", default=None)
    parser.add_argument("--verbose", default=True)
    return vars(parser.parse_args())

def main():
    args = parse_args()
    
    if args["model_name"] is None:
        raise ValueError("(finetuned) model_name must be provided.")
    if args["question"] is None:
        raise ValueError("question must be provided.")
    
    output = run_model(
        model_name=args["model_name"],
        question=args["question"],
        docs_path=args["docs_path"],
        doc_id=args["doc_id"],
        verbose=args["verbose"],
    )

if __name__ == '__main__':
    main()