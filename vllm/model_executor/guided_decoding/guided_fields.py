# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
<<<<<<< HEAD
from typing import Dict, List, Optional, TypedDict, Union
=======
from typing import Optional, TypedDict, Union
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

from pydantic import BaseModel


# These classes are deprecated, see SamplingParams
class LLMGuidedOptions(TypedDict, total=False):
<<<<<<< HEAD
    guided_json: Union[Dict, BaseModel, str]
    guided_regex: str
    guided_choice: List[str]
=======
    guided_json: Union[dict, BaseModel, str]
    guided_regex: str
    guided_choice: list[str]
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    guided_grammar: str
    guided_decoding_backend: str
    guided_whitespace_pattern: str
    guided_json_object: bool


@dataclass
class GuidedDecodingRequest:
    """One of the fields will be used to retrieve the logit processor."""
<<<<<<< HEAD
    guided_json: Optional[Union[Dict, BaseModel, str]] = None
    guided_regex: Optional[str] = None
    guided_choice: Optional[List[str]] = None
=======
    guided_json: Optional[Union[dict, BaseModel, str]] = None
    guided_regex: Optional[str] = None
    guided_choice: Optional[list[str]] = None
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    guided_grammar: Optional[str] = None
    guided_decoding_backend: Optional[str] = None
    guided_whitespace_pattern: Optional[str] = None
    guided_json_object: Optional[bool] = None
<<<<<<< HEAD

    def __post_init__(self):
        """Validate that some fields are mutually exclusive."""
        guide_count = sum([
            self.guided_json is not None, self.guided_regex is not None,
            self.guided_choice is not None, self.guided_grammar is not None,
            self.guided_json_object is not None
        ])
=======
    structural_tag: Optional[str] = None

    def __post_init__(self):
        """Validate that some fields are mutually exclusive."""
        guide_count = sum(x is not None
                          for x in (self.guided_json, self.guided_regex,
                                    self.guided_choice, self.guided_grammar,
                                    self.guided_json_object,
                                    self.structural_tag))
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        if guide_count > 1:
            raise ValueError(
                "You can only use one kind of guided decoding but multiple are "
                f"specified: {self.__dict__}")
