from llama import run_prompt_engineer_answers
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q_dirpath", default=None)
    parser.add_argument("--docs_path", default="data/docs.csv")
    parser.add_argument("--qa_path", default=None)
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--answer_system_prompt", default=None)
    parser.add_argument("--answer_prompt_suffix", default=None)
    parser.add_argument("--start_index", default=0)
    parser.add_argument("--save", default=True)
    parser.add_argument("--verbose", default=True)
    return vars(parser.parse_args())

def main():
    args = parse_args()
    if args["q_dirpath"] is None:
        raise ValueError("q_dirpath must be provided.")

    print("Generating answers for the questions provided...")
    answers = run_prompt_engineer_answers(
        args["q_dirpath"],
        docs_path=args["docs_path"],
        qa_path=args["qa_path"],
        model_name=args["model_name"],
        system_prompt=args["answer_system_prompt"],
        prompt_suffix=args["answer_prompt_suffix"],
        start_index=args["start_index"],
        save=args["save"],
        verbose=args["verbose"],
    )

    print("Done. Answers generated.")

if __name__ == '__main__':
    main()