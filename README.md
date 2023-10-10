# Docs to QA - Get a custom LLM to chat about your documents using [Lamini](https://lamini.ai) & Llama 2

Here's a quick example we've included that creates a Q&A LLM that runs on raw [legislative data (U.S. Bills)](https://huggingface.co/datasets/hyperdemocracy/us-congress-bills), without needing to manually label them yourself. You can customize this to your own dataset, below.

Here's an excerpt of what those documents in the data look like. They are not well-formatted, just raw data.
```
 118th CONGRESS 
 1st Session 
 H. CON. RES. 1 
 IN THE HOUSE OF REPRESENTATIVES 
 CONCURRENT RESOLUTION 
 Regarding consent to assemble outside the seat of government. 
 That pursuant to clause 4, section 5, article I of the Constitution, during the One Hundred Eighteenth Congress the Speaker of the House and the Majority Leader of the Senate or their respective designees, acting jointly after consultation with the Minority Leader of the House and the Minority Leader of the Senate, may notify the Members of the House and the Senate, respectively, to assemble at a place outside the District of Columbia if, in their opinion, the public interest shall warrant it.     Passed the House of Representatives January 9, 2023. Cheryl L. Johnson, Clerk."
```

From these, you can get a finetuned LLM that can intelligently answer questions about it! The LLM is customized to the types of questions you expect to ask it, and the kinds of answers you expect it to give. Give it a spin on an LLM that we finetuned for you on this data:
```
./run.sh
--question "When did H. CON. RES. 1 pass?" \
--model_name "0eb43acdd0d81f06647dfe81a1033740255c6138cc8e0a816f1308e3c784cbb9"
```

Here is the LLM's answer:
```
============MODEL ANSWER================
 Based solely on the reference material provided, H. CON. RES. 1 passed on January 9, 2023.
```

Below, we share how to use this library on your own data!

tl;dr:
1. Input a set of raw documents, i.e. just strings, a CSV, etc. We have an example with [public U.S. Bills data](https://huggingface.co/datasets/hyperdemocracy/us-congress-bills).
2. Generate questions & answers about your raw documents, by just prompt-engineering an LLM.
3. Finetune a custom Q&A LLM on this data - now it can chat over your documents, run it on new questions!
4. Hook your custom LLM up to your own application with a simple [REST API](https://lamini-ai.github.io/API/completions/).

## 1. Generate Questions
To generate the questions, run the following:

```bash
./generate_questions.sh \
--docs_dirpath "data_small" # <path_to_docs_directory>
```

All you need to do is prompt-engineer an LLM to generate questions, then answers, for you. When you run [`generate_questions.sh`](/generate_questions.sh), you can see the prompt to generate questions (`SYSTEM PROMPT`) and the questions about a docs snippet (`GENERATED QUESTION`), here a list of questions:

```
============SYSTEM PROMPT=================
You are a focused assistant who only asks questions, no chit chat. Always ask questions as helpfully as possible, while being safe. You only ask factually coherent questions about the reference text. Do not repeat the request and do not express thanks, just start asking questions and only ask questions.
============GENERATED QUESTION================
['What is the title of the document being referred to in the text?', 'What is the section of the United States Code that the text references?', 'What is the purpose of the resolution mentioned in the text?', 'Who are the individuals authorized to notify members of the House and Senate to assemble outside the District of Columbia, according to the text?', 'What is the date on which the resolution was passed by the House of Representatives, according to the text?']
```
By default, the questions generated are saved to the [`outputs/`](/outputs) folder.


### Your own prompt
Don't like the generated questions? Have a specific style you’d like the LLM to follow?

You can prompt-engineer the question-generating LLM to better match the types of questions that you expect your users to ask. Experiment with prompts and override our default prompts, as follows:

```bash
./generate_questions.sh \
--question_system_prompt "You are an expert." \
--question_prompt_suffix "Write a question:"
```

The `question_system_prompt` is the persona that you tell the LLM to assume when generating questions. The `question_prompt_suffix` can be optionally used as more context to the LLM after it reads your documents, just before it generates the questions.

### Your own data
We've included a couple default datasets in the [`data_small`](/data_small) and [`data`](/data) folders: [`data_small/docs_small.csv`](data_small/docs_small.csv) with 1 example and [`data/docs.csv`](data/docs.csv) with 100 examples for you to experiment with. 

Set the dataset using the following. Try using your own (we support csv, json, jsonl, txt, and html files)!

```bash
./generate_questions.sh \
--docs_dirpath "data_small" # <path_to_docs_directory>
```

Of course, you can provide an arbitrary amount of text -- it will just take time for the LLM to process. We recommend starting small as you prompt-engineer, so iteration speed is fast. Then, expand that dataset, if you like what you see.

### Improve further with your own Q&A examples
One way to give the LLM a flavor of the type of questions you'd like, beyond just prompt-engineering, is to provide examples of questions and answers. This can give a big boost in performance.

You can pass in a CSV file here, with `Question` as a column name:

```bash
./generate_questions.sh \
--qa_path "<path_to_csv_file>"
```

It's `qa_path` because you can pass in question-answer pairs (with column names `Question` and `Answer`) or just rows of questions (column name `Question`).

Now that you’ve generated questions, the LLM needs answers! You can apply a similar process to prompt-engineering answers to generate answers to your questions, and also providing additional Q&A data pairs for the LLM to model after.

## 2. Generate Answers
Go ahead and generate answers to your questions with the below. Just pass in the questions folder you would like to read from (this folder includes both the generated questions you had previously and the question prompt you used to get it). Of course, you can also edit the questions directly if you have other methods of improving them.

```bash
./generate_answers.sh \
--questions_dirpath "outputs/questions_20231002_005304" # <path_to_saved_questions_folder>
```

Sample output:
```
...
What are the three peaceful transfers of power mentioned in the reference text?
Answer the above question, based solely on the reference material above: [/INST]
============SYSTEM PROMPT=================
You are an expert. You answer questions factually, grounded in given reference material. Answer concisely.
=============GENERATED ANSWER================
 Based solely on the reference material above, the three peaceful transfers of power mentioned are:
1. The democratic Presidential elections in Taiwan that have yielded three peaceful transfers of power.
2. The successive parliamentary elections in Taiwan.
3. The numerous local elections in Taiwan.
```

By default, this will also save the answers generated to the [`outputs/`](/outputs/) folder.

### Prompt-Engineer Answers
Just like question generation, you can set your documents path, prompts, etc.:

```bash
./generate_answers.sh \
--questions_dirpath "<path_to_questions_folder>" \
--answer_system_prompt "You are an expert." \
--answer_prompt_suffix "Write an answer:"
```

Of course, you can run this with all arguments at once too:

```bash
./generate_answers.sh \
--questions_dirpath "<path_to_questions_folder>" \
--docs_dirpath "<path_to_docs_directory>" \
--qa_path "<path_to_csv_file>" \
--answer_system_prompt "You are an expert." \
--answer_prompt_suffix "Write answer:"
```

## 3. Finetune Your Question-Answer LLM

Now that you have question-answer pairs, it’s time to finetune (a form of LLM training)! You can run the below:

```bash
./train.sh \
--qa_dirpath "<path_to_questions_and_answers_folder>" \
--docs_dirpath "<path_to_docs_directory>"
```

You will see a model ID printed and also added to your finetuning [`Train` dashboard](https://app.lamini.ai/train), where you can track the job's status:
```
Model ID: 0eb43acdd0d81f06647dfe81a1033740255c6138cc8e0a816f1308e3c784cbb9
```

You can also set your model to be shareable with all your colleagues and friends, by passing in `is_public` as True:

```bash
./train.sh \
--qa_dirpath "<path_to_questions_and_answers_folder>" \
--is_public True
```

Lastly, you can run the finetuned Q&A LLM on a new question:

```bash
./run.sh \
--question "When did H. CON. RES. 1 pass?" \
--model_name "0eb43acdd0d81f06647dfe81a1033740255c6138cc8e0a816f1308e3c784cbb9" # <model_name>
```

