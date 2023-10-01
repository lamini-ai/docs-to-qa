# Docs to QA - Get a custom LLM to chat about your documents using [Lamini](https://lamini.ai) & Llama 2

TLDR:
1. Input a set of documents to generate questions from
2. Generate answers for the questions using your documents
3. Finetune an LLM to be able to answer questions related to your docs
4. Run your custom LLM on new questions!

## 1. Generate Questions
To generate the questions, run the following:

```bash
./generate-questions.sh
```

We have default documents available in the `data` folder. By default, the questions generated are saved to an `outputs` folder. We also provide ability to customize your LLM as follows:

### Set your own data
You can also set your own data using the following:

```bash
./generate-questions.sh
--docs_path "<path to csv file>"
```

### Set your own prompt
Have a specific style you’d like the LLM to follow? You can prompt the model using the below:


```bash
./generate-questions.sh
--question_system_prompt “You are an expert.”
--question_prompt_suffix “Write a question:”
```

### Give your own QA examples
You can even provide examples of questions and answers:

```bash
./generate-questions.sh
--qa_path "<path to csv file>"
```

Now that you’ve generated questions, the LLM needs answers! 

## 2. Generate Answers
You can generate some with the below:


```bash
./generate-answers.sh
--q_dirpath “<path to questions folder>”
```
By default this will also save the answers generated to the `outputs` folder.

### Customization
Similarly to question generation, you can set your documents path, prompts, etc.:

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

You can also set your model to be shareable:

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

