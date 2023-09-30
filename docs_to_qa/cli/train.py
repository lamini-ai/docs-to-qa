from docs_to_qa import DocsToQA
import argparse

def load_model(docs_path=None, qa_path=None, model_name=None):
    """
    Load DocsToQA model with specified 
    docs_path - default path "data/docs.csv", 
    qa_path (Optional) - default None
    and model_name - default "meta-llama/Llama-2-13b-chat-hf"
    """
    if docs_path is None:
        docs_path = "data/docs.csv"
    if model_name is None:
        model_name = "meta-llama/Llama-2-13b-chat-hf"
    llm = DocsToQA(docs_path, qa_path, model_name)
    return llm

def run_prompt_engineer_questions(
        docs_path=None,
        qa_path=None,
        model_name=None,
        system_prompt=None,
        prompt_suffix=None,
        start_index=0,
        save=True,
        verbose=True
    ):
    """
    Generates questions for each document in docs_path using an LLM.

    Input:
        Params for loading an LLM:
        docs_path (str): path to docs.csv
        qa_path (str): path to qa.csv
        model_name (str): name of model
        
        Params for generating prompt engineered questions:
        system_prompt (str): system prompt
        prompt_suffix (str): prompt suffix
        start_index (int): start index
        save (bool): whether to save
        verbose (bool): whether to print

    Output:
        questions (list): list of questions

    """
    llm = load_model(docs_path, qa_path, model_name)
    questions = llm.prompt_engineer_questions(
        system_prompt=system_prompt,
        prompt_suffix=prompt_suffix,
        start_index=start_index,
        save=save,
        verbose=verbose,
    )

    return questions
    

# Note: For good quality finetuned model, manually edit answers in answers file
# TODO: create LLM pipeline to filter answers
def run_prompt_engineer_answers(
        questions_dirpath,
        docs_path=None,
        qa_path=None,
        model_name=None,
        system_prompt=None,
        prompt_suffix=None,
        start_index=0,
        save=True,
        verbose=True
    ):
    """
    Generates answers for each question in questions_dirpath using an LLM.

    Input:
        Params for loading an LLM:
        docs_path (str, , optional): path to docs.csv
        qa_path (str, , optional): path to qa.csv
        model_name (str, , optional): name of model
        
        Params for generating prompt engineered answers:
        questions_dirpath (str): path to questions directory
        system_prompt (str, , optional): system prompt
        prompt_suffix (str, , optional): prompt suffix
        start_index (int, optional): start index
        save (bool, optional): whether to save
        verbose (bool, optional): whether to print

    Output:
        answers (list): list of answers
    """
    llm = load_model(docs_path, qa_path, model_name)
    llm.load_questions(dirpath=questions_dirpath)
    answers = llm.prompt_engineer_answers(
        system_prompt=system_prompt,
        prompt_suffix=prompt_suffix,
        start_index=start_index,
        save=save,
        verbose=verbose,
    )

    return answers

def finetune_qa(qa_dirpath, docs_path=None, model_name=None, is_public=False):
    """
    Finetunes an LLM on a set of questions and answers.

    Input:
        qa_dirpath (str): path to qa directory
        docs_path (str, optional): path to docs csv file
        model_name (str, optional): name of model
        is_public (bool, optional): whether to use public or private model

    Output:
        llm (DocsToQA): finetuned LLM
    """
    
    llm = load_model(docs_path, model_name=model_name)
    llm.load_qa(dirpath=qa_dirpath)
    llm.train(is_public=is_public)
    return llm

def run_model(model_name, question, doc_id=None, verbose=False):
    """
    Generates an answer for a question using an LLM.

    Input:
        model_name (str): name of model
        question (str): question
        doc_id (str, optional): id of document
        verbose (bool, optional): whether to print

    Output:
        output (str): answer
    """
    llm = load_model(model_name=model_name)
    output = llm.run(question, doc_id, verbose=verbose)
    if verbose:
        print("============MODEL ANSWER================")
        print(output)
    return output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_path", default="data/docs.csv")
    parser.add_argument("--qa_path", default="data/qa.csv")
    parser.add_argument("--model_name", default="meta-llama/Llama-2-13b-chat-hf")
    parser.add_argument("--question_system_prompt", default=None)
    parser.add_argument("--question_prompt_suffix", default=None)
    parser.add_argument("--answer_system_prompt", default=None)
    parser.add_argument("--answer_prompt_suffix", default=None)
    parser.add_argument("--start_index", default=0)
    parser.add_argument("--save", default=True)
    parser.add_argument("--verbose", default=True)
    parser.add_argument("--finetune", default=False)
    parser.add_argument("--question", default=None)
    parser.add_argument("--doc_id", default=None)
    parser.add_argument("qa_dirpath", default=None)

def main():
    args = parse_args()
    docs_path = args.docs_path
    qa_path = args.qa_path
    model_name = args.model_name
    
    
    # generate questions, display the first couple
    questions = run_prompt_engineer_questions(
        docs_path=args["docs_path"],
        qa_path=args["qa_path"],
        system_prompt=args["question_system_prompt"],
        prompt_suffix=args["question_prompt_suffix"],
        start_index=args["start_index"],
        save=args["save"],
        verbose=args["verbose"],
    )

    # generate answers, display the first couple
    answers = run_prompt_engineer_answers(
        questions_dirpath,
        docs_path=docs_path,
        qa_path=qa_path,
        system_prompt=args["answer_system_prompt"],
        prompt_suffix=args["answer_prompt_suffix"],
        start_index=args["start_index"],
        save=args["save"],
        verbose=args["verbose"],
    )

    # finetune the QA model
    finetuned_llm = finetune_qa(
        qa_dirpath, 
        model_name=args["base_model_name"], 
        is_public=True
    )

    # run inference
    print(run_model(
        model_name=model_name,
        question=args["question"],
        doc_id=args["doc_id"],
        verbose=True
    ))



