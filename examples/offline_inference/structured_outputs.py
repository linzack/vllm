# SPDX-License-Identifier: Apache-2.0
<<<<<<< HEAD
=======
"""
This file demonstrates the example usage of guided decoding 
to generate structured outputs using vLLM. It shows how to apply 
different guided decoding techniques such as Choice, Regex, JSON schema, 
and Grammar to produce structured and formatted results 
based on specific prompts.
"""
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

from enum import Enum

from pydantic import BaseModel

from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

<<<<<<< HEAD
llm = LLM(model="Qwen/Qwen2.5-3B-Instruct", max_model_len=100)

# Guided decoding by Choice (list of possible options)
guided_decoding_params = GuidedDecodingParams(choice=["Positive", "Negative"])
sampling_params = SamplingParams(guided_decoding=guided_decoding_params)
outputs = llm.generate(
    prompts="Classify this sentiment: vLLM is wonderful!",
    sampling_params=sampling_params,
)
print(outputs[0].outputs[0].text)

# Guided decoding by Regex
guided_decoding_params = GuidedDecodingParams(regex="\w+@\w+\.com\n")
sampling_params = SamplingParams(guided_decoding=guided_decoding_params,
                                 stop=["\n"])
prompt = ("Generate an email address for Alan Turing, who works in Enigma."
          "End in .com and new line. Example result:"
          "alan.turing@enigma.com\n")
outputs = llm.generate(prompts=prompt, sampling_params=sampling_params)
print(outputs[0].outputs[0].text)
=======
# Guided decoding by Choice (list of possible options)
guided_decoding_params_choice = GuidedDecodingParams(
    choice=["Positive", "Negative"])
sampling_params_choice = SamplingParams(
    guided_decoding=guided_decoding_params_choice)
prompt_choice = "Classify this sentiment: vLLM is wonderful!"

# Guided decoding by Regex
guided_decoding_params_regex = GuidedDecodingParams(regex=r"\w+@\w+\.com\n")
sampling_params_regex = SamplingParams(
    guided_decoding=guided_decoding_params_regex, stop=["\n"])
prompt_regex = (
    "Generate an email address for Alan Turing, who works in Enigma."
    "End in .com and new line. Example result:"
    "alan.turing@enigma.com\n")
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea


# Guided decoding by JSON using Pydantic schema
class CarType(str, Enum):
    sedan = "sedan"
    suv = "SUV"
    truck = "Truck"
    coupe = "Coupe"


class CarDescription(BaseModel):
    brand: str
    model: str
    car_type: CarType


json_schema = CarDescription.model_json_schema()
<<<<<<< HEAD

guided_decoding_params = GuidedDecodingParams(json=json_schema)
sampling_params = SamplingParams(guided_decoding=guided_decoding_params)
prompt = ("Generate a JSON with the brand, model and car_type of"
          "the most iconic car from the 90's")
outputs = llm.generate(
    prompts=prompt,
    sampling_params=sampling_params,
)
print(outputs[0].outputs[0].text)

# Guided decoding by Grammar
simplified_sql_grammar = """
    ?start: select_statement

    ?select_statement: "SELECT " column_list " FROM " table_name

    ?column_list: column_name ("," column_name)*

    ?table_name: identifier

    ?column_name: identifier

    ?identifier: /[a-zA-Z_][a-zA-Z0-9_]*/
"""
guided_decoding_params = GuidedDecodingParams(grammar=simplified_sql_grammar)
sampling_params = SamplingParams(guided_decoding=guided_decoding_params)
prompt = ("Generate an SQL query to show the 'username' and 'email'"
          "from the 'users' table.")
outputs = llm.generate(
    prompts=prompt,
    sampling_params=sampling_params,
)
print(outputs[0].outputs[0].text)
=======
guided_decoding_params_json = GuidedDecodingParams(json=json_schema)
sampling_params_json = SamplingParams(
    guided_decoding=guided_decoding_params_json)
prompt_json = ("Generate a JSON with the brand, model and car_type of"
               "the most iconic car from the 90's")

# Guided decoding by Grammar
simplified_sql_grammar = """
root ::= select_statement
select_statement ::= "SELECT " column " from " table " where " condition
column ::= "col_1 " | "col_2 "
table ::= "table_1 " | "table_2 "
condition ::= column "= " number
number ::= "1 " | "2 "
"""
guided_decoding_params_grammar = GuidedDecodingParams(
    grammar=simplified_sql_grammar)
sampling_params_grammar = SamplingParams(
    guided_decoding=guided_decoding_params_grammar)
prompt_grammar = ("Generate an SQL query to show the 'username' and 'email'"
                  "from the 'users' table.")


def format_output(title: str, output: str):
    print(f"{'-' * 50}\n{title}: {output}\n{'-' * 50}")


def generate_output(prompt: str, sampling_params: SamplingParams, llm: LLM):
    outputs = llm.generate(prompts=prompt, sampling_params=sampling_params)
    return outputs[0].outputs[0].text


def main():
    llm = LLM(model="Qwen/Qwen2.5-3B-Instruct", max_model_len=100)

    choice_output = generate_output(prompt_choice, sampling_params_choice, llm)
    format_output("Guided decoding by Choice", choice_output)

    regex_output = generate_output(prompt_regex, sampling_params_regex, llm)
    format_output("Guided decoding by Regex", regex_output)

    json_output = generate_output(prompt_json, sampling_params_json, llm)
    format_output("Guided decoding by JSON", json_output)

    grammar_output = generate_output(prompt_grammar, sampling_params_grammar,
                                     llm)
    format_output("Guided decoding by Grammar", grammar_output)


if __name__ == "__main__":
    main()
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
