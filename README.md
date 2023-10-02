# Docs to QA - Get a custom LLM to chat about your documents using [Lamini](https://lamini.ai) & Llama 2

Here's a quick example we've included that creates a Q&A LLM that runs on raw legislative data (U.S. Bills), without needing to manually label them yourself. You can customize this to your own dataset, below.

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
--question "When did H. CON. RES. 1 pass?"
--model_name "0eb43acdd0d81f06647dfe81a1033740255c6138cc8e0a816f1308e3c784cbb9"
```

Here is the LLM's answer:
```
============MODEL ANSWER================
 Based solely on the reference material provided, H. CON. RES. 1 passed on January 9, 2023.
```

Below, we share how to use this library on your own data!
TLDR:
1. Input a set of raw documents, i.e. just strings, a CSV, etc. We have an example with U.S. Bills data
2. Generate questions and answers about your documents, by just prompt-engineering an LLM
3. Finetune a Q&A LLM for your docs on this data
4. Run your custom LLM on new questions!


## 1. Generate Questions
To generate the questions, run the following:

```bash
./generate_questions.sh
```

All you need to do is prompt-engineer an LLM to generate questions, then answers, for you. When you run `generate_question.sh`, you can see the prompt to generate questions (`SYSTEM PROMPT`) and the questions about a docs snippet (`GENERATED QUESTION`), here a list of questions:

```
============SYSTEM PROMPT=================
You are a focused assistant who only asks questions, no chit chat. Always ask questions as helpfully as possible, while being safe. You only ask factually coherent questions about the reference text. Do not repeat the request and do not express thanks, just start asking questions and only ask questions.
============GENERATED QUESTION================
['What is the title of the document being referred to in the text?', 'What is the section of the United States Code that the text references?', 'What is the purpose of the resolution mentioned in the text?', 'Who are the individuals authorized to notify members of the House and Senate to assemble outside the District of Columbia, according to the text?', 'What is the date on which the resolution was passed by the House of Representatives, according to the text?']
```

Don't like the generated questions? You can prompt-engineer the system prompt to generate better questions that match your type of document, and apply it to a large number of documents. We've includes a couple default datasets in the `data` folder: `data/docs_small.csv` with 20 examples and `data/docs.csv` with 1381 examples for you to experiment with. 

Of course, you can provide an arbitrary amount of text -- it will just take time for the LLM to process. We recommend starting small to prompt-engineer, and expanding as you like what you see.

By default, the questions generated are saved to the `outputs` folder. Read on to customize this pipeline for your use case.

### Your own data
You can also set your own dataset using the following:

```bash
./generate-questions.sh
--docs_path "<path to csv file>"
```

### Your own prompt
Have a specific style you’d like the LLM to follow? You can prompt-engineer the LLM and override the default prompts, as follows:

```bash
./generate-questions.sh
--question_system_prompt “You are an expert.”
--question_prompt_suffix “Write a question:”
```

The `question_system_prompt` is the persona that you tell the LLM to assume when generating questions. The `question_prompt_suffix` can be optionally used as more context to the LLM after it reads your documents, just before it generates the questions.

### Improve further with your own QA examples
One way to give the LLM a flavor of the type of questions you'd like, beyond just prompt-engineering, is to provide examples of questions and answers. This can give a big boost in performance.

You can pass in a CSV file here:

```bash
./generate-questions.sh
--qa_path "<path to csv file>"
```

Now that you’ve generated questions, the LLM needs answers! You can apply a similar process to prompt-engineering answers to generate answers to your questions, and also providing additional Q&A data pairs for the LLM to model after.

## 2. Generate Answers
Go ahead and generate answers to your questions with the below. Just pass in the questions folder you would like to read from (this folder includes both the generated questions you had previously and the question prompt you used to get it). Of course, you can also edit the questions directly if you have other methods of improving them.

```bash
./generate-answers.sh
--q_dirpath “<path to saved questions folder>”
```

By default this will also save the answers generated to the `outputs` folder.

### Prompt-Engineer Answers
Just like question generation, you can set your documents path, prompts, etc.:

```bash
./generate-answers.sh
--questions_dirpath “<path to questions folder>”
--docs_path "<path to csv file>"
--qa_path "<path to csv file>"
--question_system_prompt “You are an expert.”
--question_prompt_suffix “Write a question:”
```

## 3. Finetune Your Question-Answer LLM

Now that you have question-answer pairs, it’s time to train! You can run the below:

```bash
./train.sh
--qa_dirpath “<path to question and answers folder>”
```

### Customization
To specify which documents to use for training:

```bash
./train.sh
--qa_dirpath “<path to question and answers folder>”
--docs_path "<path to csv file>"
```

You can also set your model to be shareable with all your colleagues and friends, by passing in `is_public` as True:

```bash
./train.sh
--qa_dirpath “<path to question and answers folder>”
--is_public True
```
Lastly, you can run the model on a question with the below:

```bash
./run.sh
--question “When did H. CON. RES. 1 pass?”
--model_name “<model name>”
```

