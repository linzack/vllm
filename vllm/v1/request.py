# SPDX-License-Identifier: Apache-2.0

import enum
<<<<<<< HEAD
from typing import TYPE_CHECKING, List, Optional, Union

from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import (EngineCoreEvent, EngineCoreEventType,
                            EngineCoreRequest, FinishReason)
from vllm.v1.utils import ConstantList

if TYPE_CHECKING:
    from vllm.multimodal import MultiModalKwargs
    from vllm.multimodal.inputs import PlaceholderRange
=======
from typing import TYPE_CHECKING, Any, Optional, Union

from vllm.multimodal.inputs import MultiModalKwargs, PlaceholderRange
from vllm.sampling_params import SamplingParams
from vllm.utils import is_list_of
from vllm.v1.engine import (EngineCoreEvent, EngineCoreEventType,
                            EngineCoreRequest, FinishReason)
from vllm.v1.structured_output.request import StructuredOutputRequest
from vllm.v1.utils import ConstantList

if TYPE_CHECKING:
    from vllm.lora.request import LoRARequest
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea


class Request:

    def __init__(
        self,
        request_id: str,
<<<<<<< HEAD
        prompt: Optional[str],
        prompt_token_ids: List[int],
        multi_modal_inputs: Optional[List["MultiModalKwargs"]],
        multi_modal_hashes: Optional[List[str]],
        multi_modal_placeholders: Optional[List["PlaceholderRange"]],
        sampling_params: SamplingParams,
        eos_token_id: Optional[int],
        arrival_time: float,
        lora_request: Optional[LoRARequest] = None,
=======
        prompt_token_ids: list[int],
        multi_modal_inputs: Optional[list[MultiModalKwargs]],
        multi_modal_hashes: Optional[list[str]],
        multi_modal_placeholders: Optional[list[PlaceholderRange]],
        sampling_params: SamplingParams,
        eos_token_id: Optional[int],
        arrival_time: float,
        lora_request: Optional["LoRARequest"] = None,
        structured_output_request: Optional["StructuredOutputRequest"] = None,
        cache_salt: Optional[str] = None,
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    ) -> None:
        self.request_id = request_id
        self.sampling_params = sampling_params
        # Because of LoRA, the eos token id can be different for each request.
        self.eos_token_id = eos_token_id
        self.lora_request = lora_request
<<<<<<< HEAD

        self.status = RequestStatus.WAITING
        self.events: List[EngineCoreEvent] = []
=======
        self.structured_output_request = structured_output_request

        self.status = (RequestStatus.WAITING_FOR_FSM
                       if sampling_params.guided_decoding is not None else
                       RequestStatus.WAITING)
        self.events: list[EngineCoreEvent] = []
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        self.stop_reason: Union[int, str, None] = None
        assert sampling_params.max_tokens is not None
        self.max_tokens = sampling_params.max_tokens

<<<<<<< HEAD
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.num_prompt_tokens = len(self.prompt_token_ids)
        self._output_token_ids: List[int] = []
        self._all_token_ids: List[int] = self.prompt_token_ids.copy()
        self.spec_token_ids: List[int] = []
        self.num_computed_tokens = 0
=======
        self.prompt_token_ids = prompt_token_ids
        self.num_prompt_tokens = len(self.prompt_token_ids)
        self._output_token_ids: list[int] = []
        self._all_token_ids: list[int] = self.prompt_token_ids.copy()
        self.spec_token_ids: list[int] = []
        self.num_computed_tokens = 0
        self.cache_salt: Optional[str] = cache_salt
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

        # Multi-modal related
        self.mm_positions = multi_modal_placeholders or []
        self.mm_inputs = multi_modal_inputs or []
<<<<<<< HEAD
        self.mm_hashes: List[str] = multi_modal_hashes or []
=======
        self.mm_hashes: list[str] = multi_modal_hashes or []
        self.num_encoder_inputs = len(self.mm_inputs)
        self.has_encoder_inputs = self.num_encoder_inputs > 0

        # P/D: Connector-specific KV transfer parameters.
        kv_params = (None if sampling_params.extra_args is None else
                     sampling_params.extra_args.get("kv_transfer_params"))
        self.kv_transfer_params: Optional[dict[str, Any]] = kv_params
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

        # Sanity check
        assert len(self.mm_inputs) == len(self.mm_positions)
        if self.mm_hashes:
            assert len(self.mm_inputs) == len(self.mm_hashes)

        # Read-only views
<<<<<<< HEAD
        # Prevent directly appending to the these lists since
=======
        # Prevent directly appending to these lists since
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        # they should also be updated simultaneously.
        self.output_token_ids = ConstantList(self._output_token_ids)
        self.all_token_ids = ConstantList(self._all_token_ids)

    @classmethod
    def from_engine_core_request(cls, request: EngineCoreRequest) -> "Request":
<<<<<<< HEAD
        return cls(
            request_id=request.request_id,
            prompt=request.prompt,
=======
        if request.mm_inputs is not None:
            assert isinstance(request.mm_inputs, list)
            assert is_list_of(request.mm_inputs, MultiModalKwargs), (
                "mm_inputs was not updated in EngineCore.add_request")

        return cls(
            request_id=request.request_id,
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
            prompt_token_ids=request.prompt_token_ids,
            multi_modal_inputs=request.mm_inputs,
            multi_modal_hashes=request.mm_hashes,
            multi_modal_placeholders=request.mm_placeholders,
            sampling_params=request.sampling_params,
            eos_token_id=request.eos_token_id,
            arrival_time=request.arrival_time,
            lora_request=request.lora_request,
<<<<<<< HEAD
        )

    def queued(self, timestamp: Optional[float] = None) -> None:
        self.events.append(
            EngineCoreEvent.new_event(EngineCoreEventType.QUEUED, timestamp))

    def scheduled(self, timestamp: Optional[float] = None) -> None:
        self.events.append(
            EngineCoreEvent.new_event(EngineCoreEventType.SCHEDULED,
                                      timestamp))

    def take_events(self) -> Optional[List[EngineCoreEvent]]:
        if not self.events:
            return None
        events, self.events = self.events, []
        return events

    def append_output_token_ids(
        self,
        token_ids: Union[int, List[int]],
    ) -> None:
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        self._output_token_ids.extend(token_ids)
        self._all_token_ids.extend(token_ids)
=======
            structured_output_request=StructuredOutputRequest(
                sampling_params=request.sampling_params),
            cache_salt=request.cache_salt,
        )

    def append_output_token_ids(
        self,
        token_ids: Union[int, list[int]],
    ) -> None:
        if isinstance(token_ids, int):
            self._output_token_ids.append(token_ids)
            self._all_token_ids.append(token_ids)
        else:
            self._output_token_ids.extend(token_ids)
            self._all_token_ids.extend(token_ids)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

    @property
    def num_tokens(self) -> int:
        return len(self._all_token_ids)

    @property
    def num_tokens_with_spec(self) -> int:
        return len(self._all_token_ids) + len(self.spec_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self._output_token_ids)

    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)

    def get_finished_reason(self) -> Union[FinishReason, None]:
        return RequestStatus.get_finished_reason(self.status)

<<<<<<< HEAD
    def has_encoder_inputs(self) -> bool:
        return len(self.mm_inputs) > 0

    @property
    def num_encoder_inputs(self) -> int:
        return len(self.mm_positions)

    def get_num_encoder_tokens(self, input_id: int) -> int:
        assert input_id < len(self.mm_positions)
        num_tokens = self.mm_positions[input_id]["length"]
        return num_tokens


class RequestStatus(enum.IntEnum):
    """Status of a request."""
    WAITING = 0
    RUNNING = 1
    PREEMPTED = 2
    # Note: anything after PREEMPTED (2) will be considered
    # as a finished status.
    FINISHED_STOPPED = 3
    FINISHED_LENGTH_CAPPED = 4
    FINISHED_ABORTED = 5
    FINISHED_IGNORED = 6
=======
    def get_num_encoder_tokens(self, input_id: int) -> int:
        assert input_id < len(self.mm_positions)
        num_tokens = self.mm_positions[input_id].length
        return num_tokens

    @property
    def use_structured_output(self) -> bool:
        return self.sampling_params.guided_decoding is not None

    def record_event(
        self,
        event_type: EngineCoreEventType,
        timestamp: Optional[float] = None,
    ) -> None:
        self.events.append(EngineCoreEvent.new_event(event_type, timestamp))

    def take_events(self) -> Optional[list[EngineCoreEvent]]:
        if not self.events:
            return None
        events, self.events = self.events, []
        return events


class RequestStatus(enum.IntEnum):
    """Status of a request."""
    WAITING = enum.auto()
    WAITING_FOR_FSM = enum.auto()
    WAITING_FOR_REMOTE_KVS = enum.auto()
    RUNNING = enum.auto()
    PREEMPTED = enum.auto()
    # Note: anything after PREEMPTED will be considered
    # as a finished status.
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_IGNORED = enum.auto()
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        return status > RequestStatus.PREEMPTED

    @staticmethod
    def get_finished_reason(
            status: "RequestStatus") -> Union[FinishReason, None]:
        return _FINISHED_REASON_MAP.get(status)


# Mapping of finished statuses to their finish reasons.
# NOTE: The ignored requests are the requests whose prompt lengths
# are longer than the model's length cap. Therefore, the stop
# reason should also be "length" as in OpenAI API.
_FINISHED_REASON_MAP = {
    RequestStatus.FINISHED_STOPPED: FinishReason.STOP,
    RequestStatus.FINISHED_LENGTH_CAPPED: FinishReason.LENGTH,
    RequestStatus.FINISHED_ABORTED: FinishReason.ABORT,
    RequestStatus.FINISHED_IGNORED: FinishReason.LENGTH,
}
