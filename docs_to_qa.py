from llama import LlamaV2Runner, BasicModelRunner
from llama.program.util.run_ai import query_run_embedding

import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
import os
import itertools
import re
import json
from collections import defaultdict
from random import sample

PROMPT_SEP = "\n"

class DocsToQA:

    def __init__(self, docs_path, qa_path=None, model_name="meta-llama/Llama-2-13b-chat-hf"):
        self.docs = {} # { "doc_id": "doc_text" }
        self.embedded_docs = {} # { "doc_id": "doc_embedding" } 
        self._chunk_docs(docs_path)
        self.qa_examples = [] # [ ["question", "answer"], ... ]
        if qa_path:
            self._get_qa_examples(qa_path)

        self.question_system_prompt = "You are a focused assistant who only asks questions, no chit chat. Always ask questions as helpfully as possible, while being safe. You only ask factually coherent questions about the reference text. Do not repeat the request and do not express thanks, just start asking questions and only ask questions."
        self.answer_system_prompt = "You are an expert. You answer questions factually, grounded in given reference material. Answer concisely."
        self.qa_system_prompt = "You are an assistant who answers questions and holds a conversation. You are helpful and friendly."

        self.question_prompt_suffix = "Write 5 questions about the above:"
        self.answer_prompt_suffix = "Answer the above question, based solely on the reference material above:"

        self.prompt_sep = PROMPT_SEP

        self.question_llm = BasicModelRunner(model_name=model_name)
        self.answer_llm = BasicModelRunner(model_name=model_name)
        self.qa_llm = BasicModelRunner(model_name=model_name)
        self.llama_prompt_template = """<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{user_prompt} [/INST]"""

        # self.question_llm = LlamaV2Runner(system_prompt=self.question_system_prompt)
        # self.answer_llm = LlamaV2Runner(system_prompt=self.answer_system_prompt)

        self.questions = defaultdict(list) # { "doc_id": ["question1", "question2", ...], ... }
        self.qa = defaultdict(list) # { "doc_id": ["question", "answer"], ... }

    def _get_qa_examples(self, qa_path):
        df = pd.read_csv(qa_path)
        for _, row in df.iterrows():
            question = row["Question"]
            answer = row["Answer"]
            self.qa_examples.append([question, answer])

    def _get_qa_prompt(self, num_examples=3):
        prompt_qa_examples = sample(self.qa_examples, min(num_examples, len(self.qa_examples)))
        prompt = "Examples of some or all task items:\n"
        for qa in prompt_qa_examples:
            question, answer = qa
            prompt += f"{question}{self.prompt_sep}{self.answer_prompt_suffix} [/INST] {answer}\n[INST] "
        prompt += "Task:\n"
        return prompt

    def _add_doc_to_question(self, question, doc_id):
        doc = self.docs[doc_id]
        question_with_doc = f"{doc}{self.prompt_sep}{question}"
        return question_with_doc

    def train(self, is_public=False): # add is_public
        # Create dataframe with columns: "question", "answer"
        rows = []
        for doc_id, qas in self.qa.items():
            for qa in qas:
                question, answer = qa
                prompt = self._make_prompt(self.qa_system_prompt, question)
                rows.append([prompt, answer])

                # Include examples with doc ("retrieval") context
                question_with_doc = self._add_doc_to_question(question, doc_id)
                prompt_with_doc = self._make_prompt(self.qa_system_prompt, question_with_doc)
                rows.append([prompt_with_doc, answer])

        df = pd.DataFrame(rows, columns=["input", "output"])
        self.qa_llm.clear_data()
        self.qa_llm.load_data_from_dataframe(df)
        self.qa_llm.train(is_public=is_public)

    def run(self, user_input, doc_id=None, verbose=False):
        if doc_id is not None:
            user_input = self._add_doc_to_question(user_input, doc_id)
        prompt = self._make_prompt(self.qa_system_prompt, user_input)
        if verbose:
            print("=============PROMPT================")
            print(prompt)
        output = self.qa_llm(prompt)
        return output

    def _get_chunks(self, text, char_chunk_size):
        chunks = []
        for i in range(0, len(text), char_chunk_size):
            chunks.append(text[i:i+char_chunk_size])
        return chunks

    def _chunk_docs(self, docs_path, char_chunk_size=5000):
        """Default chunk size is 1000""" # 5000?
        df = pd.read_csv(docs_path)
        doc_index = 0
        for _, row in df.iterrows():
            text = row["text"]
            chunks = self._get_chunks(text, char_chunk_size)
            for chunk in chunks: # TODO: discard last chunk if too small?
                self.docs[doc_index] = chunk
                doc_index += 1
                # self.docs[i] = chunk # only last chunk
                # embedding = query_run_embedding(chunk)
                # self.embedded_docs[i] = embedding

    def _make_prompt(self, system_prompt, user_prompt, cue=None):
        llama_prompt = self.llama_prompt_template.format(system_prompt=system_prompt, user_prompt=user_prompt)
        if cue:
            llama_prompt += cue
        return llama_prompt

    def _parse_enumerated_list(self, text):
        pattern = r'\d+\.\s*(.*)'
        matches = re.findall(pattern, text)
        return matches

    def _run_questions(
            self,
            start_index=0,
            save=False,
            verbose=False,
        ):
        dirpath = None
        for docs_id, chunk in tqdm(list(self.docs.items())[start_index:]):
            prompt = f"{chunk}{self.prompt_sep}{self.question_prompt_suffix}" if self.question_prompt_suffix else chunk
            if self.qa_examples:
                prompt = self._get_qa_prompt() + prompt
            prompt = self._make_prompt(self.question_system_prompt, prompt, cue="1.")
            output = self.question_llm(prompt)
            try:
                output = self._parse_enumerated_list(output)
            except:
                output = output
            self.questions[docs_id] = output
            if verbose:
                print("=============PROMPT================")
                print(prompt)
                print("============SYSTEM PROMPT=================")
                # print(question_llm.system_prompt)
                print(self.question_system_prompt)
                print("============GENERATED QUESTION================")
                print(output)
                print("=============================")
            if save:
                if dirpath:
                    self._save_questions(dirpath=dirpath, verbose=verbose)
                else:
                    dirpath = self._save_questions(verbose=verbose)
        if save:
            print(f"Saved questions to {dirpath}/questions.json")

    def load_questions(self, dirpath):
        questions_path = f"{dirpath}/questions.json"
        questions_prompt_path = f"{dirpath}/questions_prompt.json"
        self.questions = json.load(open(questions_path))
        self.questions = {int(k) if k.isdigit() else k: v for k, v in self.questions.items()}
        questions_prompt = json.load(open(questions_prompt_path))
        self.question_system_prompt = questions_prompt['system_prompt']
        self.question_prompt_suffix = questions_prompt['prompt_suffix']

    def _save_questions(self, dirpath=None, verbose=True):
        if dirpath is None:
            ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            dirpath = f"outputs/questions_{ts}"
        os.makedirs(dirpath, exist_ok=True)

        filepath = f"{dirpath}/questions.json"
        with open(filepath, 'w') as file:
            json.dump(self.questions, file)
        if verbose:
            print(f"Saved {len([question for question in self.questions.values()])} questions to {filepath}")

        prompt_filepath = f"{dirpath}/questions_prompt.json"
        with open(prompt_filepath, 'w') as file:
            json.dump({'system_prompt': self.question_system_prompt, 'prompt_suffix': self.question_prompt_suffix}, file)
        if verbose:
            print(f"Saved question prompt to {prompt_filepath}")
        return dirpath

    def prompt_engineer_questions(
            self,
            system_prompt=None,
            prompt_suffix=None,
            start_index=0,
            save=False,
            verbose=True,
        ):
        if system_prompt is not None:
            self.question_system_prompt = system_prompt
        if prompt_suffix is not None:
            self.question_prompt_suffix = prompt_suffix

        # question_llm = LlamaV2Runner(system_prompt=prompt)
        self._run_questions(start_index=start_index, save=save, verbose=verbose)

        return self.questions

    def _run_answers(
            self,
            start_index=0,
            batch_size=1,
            save=False,
            verbose=False,
        ):
        if save:
            ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            dirpath = f"outputs/qa_{ts}"
            os.makedirs(dirpath, exist_ok=True)

            self._save_questions(dirpath=dirpath)

            prompt_filepath = f"{dirpath}/answers_prompt.json"
            with open(prompt_filepath, 'w') as file:
                json.dump({'system_prompt': self.answer_system_prompt, 'prompt_suffix': self.answer_prompt_suffix}, file)
            print(f"Saved answers prompt to {prompt_filepath}")

        doc_ids_in_questions = [int(k) for k in self.questions.keys()]
        prompts_list = []
        questions_list = []
        docs_ids_list = []
        for docs_id, chunk in self.docs.items():
            if docs_id not in doc_ids_in_questions:
                continue
            question_list = self.questions[docs_id]
            for question in question_list:
                prompt = f"{chunk}{self.prompt_sep}{question}{self.prompt_sep}{self.answer_prompt_suffix}" if self.answer_prompt_suffix else f"{chunk}{self.prompt_sep}{question}"
                if self.qa_examples:
                    prompt = self._get_qa_prompt() + prompt
                prompt = self._make_prompt(self.answer_system_prompt, prompt)

                prompts_list.append(prompt)
                questions_list.append(question)
                docs_ids_list.append(docs_id)

        # Iterate through prompts in batches
        for i in tqdm(range(start_index, len(prompts_list), batch_size)):
            batch_prompts = prompts_list[i:i+batch_size]
            batch_answers = self.answer_llm(batch_prompts)
            batch_questions = questions_list[i:i+batch_size]
            batch_docs_ids = docs_ids_list[i:i+batch_size]
            for j in range(len(batch_questions)):
                question = batch_questions[j]
                prompt = batch_prompts[j]
                answer = batch_answers[j]['output']
                docs_id = batch_docs_ids[j]

                self.qa[docs_id].append([question, answer])
                if verbose:
                    print("=============PROMPT================")
                    print(prompt)
                    print("============SYSTEM PROMPT=================")
                    print(self.answer_system_prompt)
                    print("=============GENERATED ANSWER================")
                    print(answer)
                    print("=============================")
            if save:
                filepath = f"{dirpath}/qa.json"
                with open(filepath, 'w') as file:
                    json.dump(self.qa, file)
                if verbose:
                    print(f"Saved {i + batch_size} answers to {filepath}")
        if save:
            print(f"Saved answers to {filepath}")

    def load_qa(self, dirpath):
        qa_path = f"{dirpath}/qa.json"
        answers_prompt_path = f"{dirpath}/answers_prompt.json"

        self.qa = json.load(open(qa_path))
        self.qa = {int(k) if k.isdigit() else k: v for k, v in self.qa.items()}
        answers_prompt = json.load(open(answers_prompt_path))
        self.answer_system_prompt = answers_prompt['system_prompt']
        self.answer_prompt_suffix = answers_prompt['prompt_suffix']

        questions_path = f"{dirpath}/questions.json"
        questions_prompt_path = f"{dirpath}/questions_prompt.json"

        self.questions = json.load(open(questions_path))
        self.questions = {int(k) if k.isdigit() else k: v for k, v in self.questions.items()}
        questions_prompt = json.load(open(questions_prompt_path))
        self.question_system_prompt = questions_prompt['system_prompt']
        self.question_prompt_suffix = questions_prompt['prompt_suffix']

    def prompt_engineer_answers(
            self,
            system_prompt=None,
            prompt_suffix=None,
            start_index=0,
            questions=None,
            save=False,
            verbose=True,
        ):
        if system_prompt is not None:
            self.answer_system_prompt = system_prompt
        if prompt_suffix is not None:
            self.answer_prompt_suffix = prompt_suffix
        if questions is None:
            assert self.questions, "You must set questions first, or pass questions in as {doc_id: question}"
            questions = self.questions

        # answer_llm = LlamaV2Runner(system_prompt=prompt)
        self._run_answers(start_index=start_index, save=save, verbose=verbose)

        return self.qa


def save_docs():
    hf_docs_path = "hyperdemocracy/us-congress-bills"
    dataset = load_dataset(hf_docs_path, split="train", streaming=True)
    top_n = itertools.islice(dataset, 100)
    rows = []
    for i in tqdm(top_n):
        rows.append(i["text"])
    df = pd.DataFrame(rows, columns=["text"])
    df.to_csv("data/docs.csv", index=False)

def load_model(docs_path=None, qa_path=None, model_name=None):
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
    llm = load_model(docs_path, qa_path, model_name)
    llm.prompt_engineer_questions(
        system_prompt=system_prompt,
        prompt_suffix=prompt_suffix,
        start_index=start_index,
        save=save,
        verbose=verbose,
    )

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
    llm = load_model(docs_path, qa_path, model_name)
    llm.load_questions(dirpath=questions_dirpath)
    llm.prompt_engineer_answers(
        system_prompt=system_prompt,
        prompt_suffix=prompt_suffix,
        start_index=start_index,
        save=save,
        verbose=verbose,
    )
    # Manually edit answers in answers file
    # Or TODO: create LLM pipeline to filter answers

def finetune_qa(qa_dirpath, docs_path=None, model_name=None, is_public=False):
    llm = load_model(docs_path, model_name=model_name)
    llm.load_qa(dirpath=qa_dirpath)
    llm.train(is_public=is_public)

def run_model(model_name, question, doc_id=None, verbose=False):
    llm = load_model(model_name=model_name)
    output = llm.run(question, doc_id, verbose=verbose)
    if verbose:
        print("============MODEL ANSWER================")
        print(output)
    return output

def run_model_on_questions(model_name, questions_dirpath, verbose=False):
    llm = load_model(model_name=model_name)
    llm.load_questions(dirpath=questions_dirpath)
    for doc_id in llm.questions:
        for question in llm.qa[doc_id]:
            output = llm.run(question, doc_id, verbose=verbose)
            if verbose:
                print("============MODEL ANSWER================")
                print(output)

def run_model_on_qa(model_name, qa_dirpath, verbose=False):
    llm = load_model(model_name=model_name)
    llm.load_qa(dirpath=qa_dirpath)
    for doc_id in llm.qa:
        for qa in llm.qa[doc_id]:
            question, answer = qa[0], qa[1]
            output = llm.run(question, doc_id, verbose=verbose)
            if verbose:
                print("============ACTUAL ANSWER=================")
                print(answer)
                print("============MODEL ANSWER================")
                print(output)

if __name__ == "__main__":
    # # Get dataset
    # save_docs()

    # # Generate questions
    docs_path = "data/test_docs.csv"
    # docs_path = "data/docs.csv"
    qa_path = "data/qa.csv"
    question_system_prompt = "You are a focused assistant who only asks questions, no chit chat. Always ask questions as helpfully as possible, while being safe. You only ask factually coherent questions about the reference text. Do not repeat the request and do not express thanks, just start asking questions and only ask questions."
    question_prompt_suffix = "Write 5 questions about the above:"
    start_index = 0
    save = False
    # save = True
    verbose = True
    # verbose = False
    run_prompt_engineer_questions(
        docs_path=docs_path,
        qa_path=qa_path,
        system_prompt=question_system_prompt,
        prompt_suffix=question_prompt_suffix,
        start_index = start_index,
        save=save,
        verbose=verbose,
    )

    # # Generate answers
    questions_dirpath = "outputs/questions_20230927_232532"
    answer_system_prompt = "You are an expert. You answer questions factually, grounded in given reference material. Answer concisely."
    answer_prompt_suffix = "Answer the above question, based solely on the reference material above:"
    start_index = 0
    save = False
    # save = True
    verbose = True
    # verbose = False
    # run_prompt_engineer_answers(
    #     questions_dirpath,
    #     docs_path=docs_path,
    #     qa_path=qa_path,
    #     system_prompt=answer_system_prompt,
    #     prompt_suffix=answer_prompt_suffix,
    #     start_index=start_index,
    #     save=save,
    #     verbose=verbose,
    # )

    # # Finetune model
    qa_dirpath = "outputs/qa_20230928_224709"
    # base_model_name = "meta-llama/Llama-2-13b-chat-hf"
    base_model_name = "meta-llama/Llama-2-13b-hf"
    # finetune_qa(qa_dirpath, model_name=base_model_name, is_public=True)

    # Evaluate model
    # model_name = "meta-llama/Llama-2-13b-chat-hf"
    # model_name = "meta-llama/Llama-2-13b-hf"
    # model_name = "16ed90809934a2a0a8a783ab20da60d66fedc67f56e0f2ab4cdb712f10a4f569" # finetuned Llama 2 13b chat
    # model_name = "1807b1658bc3edb10de8f300b56c1f4cfc602a2a9f5c78422fe6b79097af50a2" # finetuned Llama 2 13b
    model_name = "a1ef30d0362fddc0b6ead81d3f1c5c6b1211baff81f0265a3cfcb65be188ced5" # finetuned Llama 2 13b
    question = "What is the date on which H. CON. RES. 1 was passed by the House of Representatives, according to the text?"
    # question = "When did H. CON. RES. 1 pass?"
    doc_id = 0
    # doc_id = None
    # run_model(
    #     model_name=model_name,
    #     question=question,
    #     doc_id=doc_id,
    #     verbose=True
    # )
    # run_model_on_questions(
    #     model_name=model_name,
    #     questions_dirpath=questions_dirpath,
    #     verbose=True
    # )
    # run_model_on_qa(
    #     model_name=model_name,
    #     qa_dirpath=qa_dirpath,
    #     verbose=True
    # )
