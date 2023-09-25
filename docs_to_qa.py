from llama import LlamaV2Runner, BasicModelRunner
from llama.program.util.run_ai import query_run_embedding

import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
import os
import itertools
import re
import json


class DocsToQA:

    def __init__(self, docs_path):
        self.docs = {} # { "doc_id": "doc_text" }
        self.embedded_docs = {} # { "doc_id": "doc_embedding" } 
        self._chunk_docs(docs_path)

        self.question_system_prompt = "You are a focused assistant who only asks questions, no chit chat. Always ask questions as helpfully as possible, while being safe. You only ask factually coherent questions about the reference text. Do not repeat the request and do not express thanks, just start asking questions and only ask questions."
        self.answer_system_prompt = "You are an expert. You answer questions factually, grounded in given reference material."

        self.question_llm = BasicModelRunner(model_name="meta-llama/Llama-2-13b-chat-hf")
        self.answer_llm = BasicModelRunner(model_name="meta-llama/Llama-2-13b-chat-hf")
        self.llama_prompt_template = """<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{user_prompt} [/INST]"""
        
        # self.question_llm = LlamaV2Runner(system_prompt=self.question_system_prompt)
        # self.answer_llm = LlamaV2Runner(system_prompt=self.answer_system_prompt)

        self.questions = {} # { "doc_id": "question" }
        self.answers = {} # { "doc_id": "answer" }

    def _get_chunks(self, text, char_chunk_size):
        chunks = []
        for i in range(0, len(text), char_chunk_size):
            chunks.append(text[i:i+char_chunk_size])
        return chunks

    def _chunk_docs(self, docs_path, char_chunk_size=5000):
        """Default chunk size is 1000"""
        df = pd.read_csv(docs_path)
        for i, row in df.iterrows():
            text = row["text"]
            chunks = self._get_chunks(text, char_chunk_size)
            for chunk in chunks:
                self.docs[i] = chunk
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
    
    def _run_questions(self,
                      prompt_suffix="Write 5 questions about the above:", 
                      prompt_sep="\n",
                      verbose=False):
        questions = {}
        for docs_id, chunk in tqdm(self.docs.items()):
            prompt = f"{chunk}{prompt_sep}{prompt_suffix}" if prompt_suffix else chunk
            prompt = self._make_prompt(self.question_system_prompt, prompt, cue="1.")
            output = self.question_llm(prompt)
            try:
                output = self._parse_enumerated_list(output)
            except:
                output = output
            questions[docs_id] = output
            if verbose:
                print("=============PROMPT================")
                print(prompt)
                print("============SYSTEM PROMPT=================")
                # print(question_llm.system_prompt)
                print(self.question_system_prompt)
                print("============GENERATED QUESTION================")
                print(output)
                print("=============================")
        return questions

    def load_questions(self, load_from):
        self.questions = json.load(open(load_from))
        self.question_system_prompt = open(load_from.replace(".json", "_prompt.txt")).read()
    
    def _save_questions(self, dirpath="outputs"):
        ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(dirpath, exist_ok=True)

        filepath = f"{dirpath}/questions_{ts}.json" if dirpath else f"questions_{ts}.json"
        with open(filepath, 'w') as file:
            json.dump(self.questions, file)
        print(f"Saved questions to {filepath}")

        prompt_filepath = f"{dirpath}/questions_prompt_{ts}.txt" if dirpath else f"questions_prompt_{ts}.txt"
        with open(prompt_filepath, 'w') as file:
            file.write(self.question_system_prompt)
        print(f"Saved question prompt to {prompt_filepath}")
        
    def prompt_engineer_questions(self, prompt=None, prompt_suffix=None, save=False):
        if prompt:
            self.question_system_prompt = prompt
        
        # question_llm = LlamaV2Runner(system_prompt=prompt)
        self.questions = self._run_questions(prompt_suffix=prompt_suffix, verbose=True)
        
        if save:
            self._save_questions()
        return self.questions
    
    def _run_answers(self,
                    prompt_suffix="Answer the above question, based solely on the reference material above:",
                    prompt_sep="\n",
                    verbose=False):
        answers = {}
        for docs_id, chunk in tqdm(self.docs.items()):
            if docs_id not in self.questions:
                continue
            question = self.questions[docs_id]
            prompt = f"{chunk}{prompt_sep}{question}{prompt_sep}{prompt_suffix}"
            prompt = self._make_prompt(self.answer_system_prompt, prompt, cue="Answer:")
            output = self.answer_llm(prompt)
            answers[docs_id] = output
            if verbose:
                print("=============PROMPT================")
                print(prompt)
                print("============SYSTEM PROMPT=================")
                print(self.answer_system_prompt)
                # print(answer_llm.system_prompt)
                print("=============GENERATED ANSWER================")
                print(output)
                print("=============================")
        return answers
    
    def _save_answers(self):
        ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        dirpath = f"outputs/qa_{ts}"
        os.makedirs(dirpath, exist_ok=True)

        self.save_questions(dirpath=dirpath)

        filepath = f"{dirpath}/answers_{ts}.json"
        with open(filepath, 'w') as file:
            json.dump(self.answers, file)
        print(f"Saved answers to {filepath}")
        
        prompt_filepath = f"{dirpath}/answers_prompt_{ts}.txt"
        with open(prompt_filepath, 'w') as file:
            file.write(self.answer_system_prompt)
        print(f"Saved answers prompt to {prompt_filepath}")

    def load_qa(self, load_from):
        self.answers = json.load(open(load_from))
        self.answer_system_prompt = open(load_from.replace(".json", "_prompt.txt")).read()
        self.question_system_prompt = open(load_from.replace("answers", "questions").replace(".json", "_prompt.txt")).read()
        self.questions = json.load(open(load_from.replace("answers", "questions")))
     
    def prompt_engineer_answers(self, prompt=None, prompt_suffix=None, questions=None, save=False):
        if prompt is None:
            prompt = self.answer_system_prompt
        if questions is None:
            assert self.questions, "You must set questions first, or pass questions in as {doc_id: question}"
            questions = self.questions

        # answer_llm = LlamaV2Runner(system_prompt=prompt)
        if prompt_suffix is None:
            self.answers = self._run_answers(verbose=True)
        else:
            self.answers = self._run_answers(prompt_suffix=prompt_suffix, verbose=True)
        
        if save:
            self._save_answers()
        return self.answers


def get_dataset():
    hf_docs_path = "TaylorAI/pubmed_commercial"
    dataset = load_dataset(hf_docs_path, split="train", streaming=True)
    top_n = itertools.islice(dataset, 100)
    rows = []
    for i in top_n:
        rows.append(i["text"])
    df = pd.DataFrame(rows, columns=["text"])
    df.to_csv("docs.csv", index=False)

    if not os.path.exists("docs.csv"):
        get_dataset()

def load_model():
    docs_path = "docs.csv"
    llm = DocsToQA(docs_path)

    return llm

def run_prompt_engineer_questions():
    llm = load_model()
    llm.prompt_engineer_questions(save=True)

def prompt_engineer_answers():
    llm = load_model()
    llm.load_questions(load_from="questions_20230924_210114.json")
    llm.prompt_engineer_answers()

def main():
    docs_path = "docs.csv"
    llm = DocsToQA(docs_path)

    llm.prompt_engineer_questions(save=True)
    # llm.load_questions(load_from="")
    # llm.prompt_engineer_answers()

main()