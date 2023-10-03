from llama.docs_to_qa.docs_to_qa import run_prompt_engineer_questions
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_path", default="data/docs_small.csv")
    parser.add_argument("--qa_path", default=None)
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--question_system_prompt", default=None)
    parser.add_argument("--question_prompt_suffix", default=None)
    parser.add_argument("--start_index", default=0)
    parser.add_argument("--save", default=True)
    parser.add_argument("--verbose", default=True)

    return vars(parser.parse_args())

def main():
    args = parse_args()
    print("Arguments parsed. Setting the arguments not passed to default. Arguments:")
    print(args)

    print("Generating questions...")
    # generate questions, display the first couple
    questions = run_prompt_engineer_questions(
        docs_path=args["docs_path"],
        qa_path=args["qa_path"],
        model_name=args["model_name"],
        system_prompt=args["question_system_prompt"],
        prompt_suffix=args["question_prompt_suffix"],
        start_index=args["start_index"],
        save=args["save"],
        verbose=args["verbose"],
    )

    print("Done. Questions generated.")

if __name__ == '__main__':
    main()
