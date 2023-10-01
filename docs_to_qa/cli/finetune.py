from llama import finetune_qa, run_model
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_dirpath", default=None)
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--question", default=None)
    parser.add_argument("--doc_id", default=None)
    parser.add_argument("--verbose", default=True)
    return vars(parser.parse_args())

def main():
    args = parse_args()

    if args["qa_dirpath"] is None:
        raise ValueError("qa_dirpath must be provided.")
    
    finetune_llm = finetune_qa(
        qa_dirpath=args["qa_dirpath"], 
        model_name=args["model_name"], 
        is_public=True,
    )

    print("Model Finetuned. Model name: ", finetune_llm)

    print("Generating answer for the question provided...")
    output = run_model(
        model_name=finetune_llm,
        question=args["question"],
        doc_id=args["doc_id"],
        verbose=args["verbose"],
    )


if __name__ == '__main__':
    main()