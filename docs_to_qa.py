from llama import LlamaV2Runner

import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
import os
import itertools


class DocsToQA:

    def __init__(self, docs_path):
        self.docs = {} # { "doc_id": "doc_text" }
        self.embedded_docs = {} # { "doc_id": "doc_embedding" } 
        self.chunk_docs(docs_path)

        self.question_prompt = "You are an inquisitive expert, whose job is to ask questions. You write factual questions or requests about a text."
        self.answer_prompt = "You are an expert. You answer questions factually, grounded in given reference material."

        self.question_llm = LlamaV2Runner(system_prompt=self.question_prompt)
        self.answer_llm = LlamaV2Runner(system_prompt=self.answer_prompt)

        self.questions = {} # { "doc_id": "question" }
        self.answers = {} # { "doc_id": "answer" }

    def chunk_docs(self, docs_path):
        """Default chunk size is 1000"""
        df = pd.read_csv(docs_path)
        for i, chunk in df.iterrows():
            self.docs[i] = chunk["text"]
            # embedding = query_run_embedding(chunk)
            # self.embedded_docs[i] = embedding
    
    def set_question_prompt(self, prompt):
        self.question_prompt = prompt
        self.question_llm = LlamaV2Runner(system_prompt=prompt)
        self.set_questions()
    
    def run_questions(self,
                      docs,
                      question_llm,
                      prompt_suffix="Write a question about the above:", 
                      prompt_sep="\n",
                      verbose=False):
        questions = {}
        for docs_id, chunk in tqdm(docs.items()):
            prompt = f"{chunk}{prompt_sep}{prompt_suffix}"
            output = question_llm(chunk)
            questions[docs_id] = output
            if verbose:
                print("=============PROMPT================")
                print(prompt)
                print("============SYSTEM PROMPT=================")
                print(question_llm.system_prompt)
                print("============GENERATED QUESTION================")
                print(output)
                print("=============================")
        return questions

    def set_questions(self):
        self.questions = self.run_questions(self.docs, self.question_llm)

    def prompt_engineer_questions(self, prompt, prompt_suffix=None):
        question_llm = LlamaV2Runner(system_prompt=prompt)
        if prompt_suffix:
            self.run_questions(self.docs, question_llm, prompt_suffix=prompt_suffix, verbose=True)
        else:
            self.run_questions(self.docs, question_llm, verbose=True)
    
    def set_answer_prompt(self, prompt):
        self.answer_prompt = prompt
        self.answer_llm = LlamaV2Runner(system_prompt=prompt)
        self.set_answers()
    
    def run_answers(self, docs, questions, answer_llm, prompt_sep="\n", verbose=False):
        answers = {}
        for docs_id, chunk in tqdm(docs.items()):
            if docs_id not in questions:
                continue
            question = questions[docs_id]
            prompt = f"{question}{prompt_sep}{chunk}"
            output = answer_llm(prompt)
            answers[docs_id] = output
            if verbose:
                print("=============================")
                print(answer_llm.system_prompt)
                print("=============================")
                print(prompt)
                print("=============================")
                print(output)
                print("=============================")
        return answers
    
    def set_answers(self):
        assert self.questions, "You must set questions first"
        self.answers = self.run_answers(self.docs, self.questions, self.answer_llm)
    
    def prompt_engineer_answers(self, prompt, questions=None):
        if questions is None:
            assert self.questions, "You must set questions first, or pass questions in as {doc_id: question}"
            questions = self.questions

        answer_llm = LlamaV2Runner(system_prompt=prompt)
        self.run_answers(self.docs, questions, answer_llm, verbose=True)

def get_dataset():
    hf_docs_path = "TaylorAI/pubmed_commercial"
    dataset = load_dataset(hf_docs_path, split="train", streaming=True)
    top_n = itertools.islice(dataset, 100)
    rows = []
    for i in top_n:
        rows.append(i["text"])
    df = pd.DataFrame(rows, columns=["text"])
    df.to_csv("docs.csv", index=False)

def main():
    if not os.path.exists("docs.csv"):
        get_dataset()

    docs_path = "docs.csv"
    llm = DocsToQA(docs_path)

    llm.prompt_engineer_questions(llm.question_prompt)

main()