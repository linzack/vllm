# SPDX-License-Identifier: Apache-2.0
<<<<<<< HEAD
=======
"""
To run this example, you need to start the vLLM server:

```bash
vllm serve Qwen/Qwen2.5-3B-Instruct
```
"""
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

from enum import Enum

from openai import BadRequestError, OpenAI
from pydantic import BaseModel

<<<<<<< HEAD
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="-",
)

# Guided decoding by Choice (list of possible options)
completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-3B-Instruct",
    messages=[{
        "role": "user",
        "content": "Classify this sentiment: vLLM is wonderful!"
    }],
    extra_body={"guided_choice": ["positive", "negative"]},
)
print(completion.choices[0].message.content)

# Guided decoding by Regex
prompt = ("Generate an email address for Alan Turing, who works in Enigma."
          "End in .com and new line. Example result:"
          "alan.turing@enigma.com\n")

completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-3B-Instruct",
    messages=[{
        "role": "user",
        "content": prompt,
    }],
    extra_body={
        "guided_regex": "\w+@\w+\.com\n",
        "stop": ["\n"]
    },
)
print(completion.choices[0].message.content)
=======

# Guided decoding by Choice (list of possible options)
def guided_choice_completion(client: OpenAI, model: str):
    completion = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": "Classify this sentiment: vLLM is wonderful!"
        }],
        extra_body={"guided_choice": ["positive", "negative"]},
    )
    return completion.choices[0].message.content


# Guided decoding by Regex
def guided_regex_completion(client: OpenAI, model: str):
    prompt = ("Generate an email address for Alan Turing, who works in Enigma."
              "End in .com and new line. Example result:"
              "alan.turing@enigma.com\n")

    completion = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": prompt,
        }],
        extra_body={
            "guided_regex": r"\w+@\w+\.com\n",
            "stop": ["\n"]
        },
    )
    return completion.choices[0].message.content
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


<<<<<<< HEAD
json_schema = CarDescription.model_json_schema()

prompt = ("Generate a JSON with the brand, model and car_type of"
          "the most iconic car from the 90's")
completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-3B-Instruct",
    messages=[{
        "role": "user",
        "content": prompt,
    }],
    extra_body={"guided_json": json_schema},
)
print(completion.choices[0].message.content)

# Guided decoding by Grammar
simplified_sql_grammar = """
    ?start: select_statement

    ?select_statement: "SELECT " column_list " FROM " table_name

    ?column_list: column_name ("," column_name)*

    ?table_name: identifier

    ?column_name: identifier

    ?identifier: /[a-zA-Z_][a-zA-Z0-9_]*/
"""

prompt = ("Generate an SQL query to show the 'username' and 'email'"
          "from the 'users' table.")
completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-3B-Instruct",
    messages=[{
        "role": "user",
        "content": prompt,
    }],
    extra_body={"guided_grammar": simplified_sql_grammar},
)
print(completion.choices[0].message.content)

# Extra backend options
prompt = ("Generate an email address for Alan Turing, who works in Enigma."
          "End in .com and new line. Example result:"
          "alan.turing@enigma.com\n")

try:
    # The no-fallback option forces vLLM to use xgrammar, so when it fails
    # you get a 400 with the reason why
    completion = client.chat.completions.create(
        model="Qwen/Qwen2.5-3B-Instruct",
=======
def guided_json_completion(client: OpenAI, model: str):
    json_schema = CarDescription.model_json_schema()

    prompt = ("Generate a JSON with the brand, model and car_type of"
              "the most iconic car from the 90's")
    completion = client.chat.completions.create(
        model=model,
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        messages=[{
            "role": "user",
            "content": prompt,
        }],
<<<<<<< HEAD
        extra_body={
            "guided_regex": "\w+@\w+\.com\n",
            "stop": ["\n"],
            "guided_decoding_backend": "xgrammar:no-fallback"
        },
    )
except BadRequestError as e:
    print("This error is expected:", e)
=======
        extra_body={"guided_json": json_schema},
    )
    return completion.choices[0].message.content


# Guided decoding by Grammar
def guided_grammar_completion(client: OpenAI, model: str):
    simplified_sql_grammar = """
        root ::= select_statement

        select_statement ::= "SELECT " column " from " table " where " condition

        column ::= "col_1 " | "col_2 "

        table ::= "table_1 " | "table_2 "

        condition ::= column "= " number

        number ::= "1 " | "2 "
    """

    prompt = ("Generate an SQL query to show the 'username' and 'email'"
              "from the 'users' table.")
    completion = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": prompt,
        }],
        extra_body={"guided_grammar": simplified_sql_grammar},
    )
    return completion.choices[0].message.content


# Extra backend options
def extra_backend_options_completion(client: OpenAI, model: str):
    prompt = ("Generate an email address for Alan Turing, who works in Enigma."
              "End in .com and new line. Example result:"
              "alan.turing@enigma.com\n")

    try:
        # The guided_decoding_disable_fallback option forces vLLM to use
        # xgrammar, so when it fails you get a 400 with the reason why
        completion = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": prompt,
            }],
            extra_body={
                "guided_regex": r"\w+@\w+\.com\n",
                "stop": ["\n"],
                "guided_decoding_backend": "xgrammar",
                "guided_decoding_disable_fallback": True,
            },
        )
        return completion.choices[0].message.content
    except BadRequestError as e:
        print("This error is expected:", e)


def main():
    client: OpenAI = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="-",
    )

    model = client.models.list().data[0].id

    print("Guided Choice Completion:")
    print(guided_choice_completion(client, model))

    print("\nGuided Regex Completion:")
    print(guided_regex_completion(client, model))

    print("\nGuided JSON Completion:")
    print(guided_json_completion(client, model))

    print("\nGuided Grammar Completion:")
    print(guided_grammar_completion(client, model))

    print("\nExtra Backend Options Completion:")
    print(extra_backend_options_completion(client, model))


if __name__ == "__main__":
    main()
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
