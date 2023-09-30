from docs_to_qa import run_prompt_engineer_answers
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("questions_dirpath", default=None)
    parser.add_argument("--docs_path", default="data/docs.csv")
    parser.add_argument("--qa_path", default="data/qa.csv")
    parser.add_argument("--answer_system_prompt", default=None)
    parser.add_argument("--answer_prompt_suffix", default=None)
    parser.add_argument("--start_index", default=0)
    parser.add_argument("--save", default=True)
    parser.add_argument("--verbose", default=True)


def main():

    args = parse_args()
    print("Arguments parsed. Setting the arguments not passed to default. Arguments:")
    print(args)

    print("Generating answers for the questions provided...")
    # generate questions, display the first couple
    answers = run_prompt_engineer_answers(
        args["questions_dirpath"],
        docs_path=args["docs_path"],
        qa_path=args["qa_path"],
        system_prompt=args["answer_system_prompt"],
        prompt_suffix=args["answer_prompt_suffix"],
        start_index=args["start_index"],
        save=args["save"],
        verbose=args["verbose"],
    )

    print("Done. Answers generated.")
