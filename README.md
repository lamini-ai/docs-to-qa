# Docs to QA - Get a custom LLM to chat about your documents using [Lamini](https://lamini.ai) & Llama 2

TLDR:
1. Input a set of documents to generate questions from
2. Generate answers for the questions using your documents
3. Finetune an LLM to be able to answer questions related to your docs
4. Run your custom LLM on new questions!

## 1. Generate Questions
To generate the questions, run the following:

```bash
./generate_questions.sh
```

This example runs on legislative data of U.S. Bills to create question-answer pairs related to these docs, without needing to manually label them yourself! All you need to do is prompt-engineer an LLM to generate questions, then answers, for you. When you run `generate_question.sh`, you can see the prompt to generate questions (`SYSTEM PROMPT`) and the questions about a docs snippet (`GENERATED QUESTION`), here a list of questions:

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
Have a specific style you’d like the LLM to follow? You can prompt the model using the below:

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

