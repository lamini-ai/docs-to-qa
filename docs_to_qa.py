from llama import LlamaV2Runner
from llama.program.util.run_ai import query_run_embedding

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

        self.question_prompt = "You are a focused assistant who only asks questions, no chit chat. Always ask questions as helpfully as possible, while being safe. You only ask factually coherent questions about the reference text. Do not repeat the request and do not express thanks, just start asking questions and only ask questions."
        self.answer_prompt = "You are an expert. You answer questions factually, grounded in given reference material."

        self.question_llm = LlamaV2Runner(system_prompt=self.question_prompt)
        self.answer_llm = LlamaV2Runner(system_prompt=self.answer_prompt)

        self.questions = {} # { "doc_id": "question" }
        self.answers = {} # { "doc_id": "answer" }

    def get_chunks(self, text, char_chunk_size):
        chunks = []
        for i in range(0, len(text), char_chunk_size):
            chunks.append(text[i:i+char_chunk_size])
        return chunks

    def chunk_docs(self, docs_path, char_chunk_size=5000):
        """Default chunk size is 1000"""
        df = pd.read_csv(docs_path)
        for i, row in df.iterrows():
            text = row["text"]
            chunks = self.get_chunks(text, char_chunk_size)
            for chunk in chunks:
                self.docs[i] = chunk
                # embedding = query_run_embedding(chunk)
                # self.embedded_docs[i] = embedding
    
    def set_question_prompt(self, prompt):
        self.question_prompt = prompt
        self.question_llm = LlamaV2Runner(system_prompt=prompt)
        self.set_questions()
    
    def run_set_questions(self):
        self.run_questions(self.docs, self.question_llm, verbose=True)

    def run_questions(self,
                      docs,
                      question_llm,
                      prompt_suffix="Write 5 questions about the above:", 
                      prompt_sep="\n",
                      verbose=False):
        questions = {}
        for docs_id, chunk in tqdm(docs.items()):
            chunk = chunk[:1000]
            prompt = f"{chunk}{prompt_sep}{prompt_suffix}"
            output = question_llm(prompt)
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

    llm.run_set_questions()

main()