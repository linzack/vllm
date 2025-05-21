# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
<<<<<<< HEAD
from typing import Any, Optional, Set
=======
from typing import Any, Optional
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

import torch


class AbstractWorkerManager(ABC):

    def __init__(self, device: torch.device):
        self.device = device

    @property
    @abstractmethod
    def is_enabled(self) -> bool:
        raise NotImplementedError

    @abstractmethod
<<<<<<< HEAD
    def set_active_adapters(self, requests: Set[Any],
=======
    def set_active_adapters(self, requests: set[Any],
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
                            mapping: Optional[Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def add_adapter(self, adapter_request: Any) -> bool:
        raise NotImplementedError

    @abstractmethod
    def remove_adapter(self, adapter_id: int) -> bool:
        raise NotImplementedError

    @abstractmethod
    def remove_all_adapters(self) -> None:
        raise NotImplementedError

    @abstractmethod
<<<<<<< HEAD
    def list_adapters(self) -> Set[int]:
=======
    def list_adapters(self) -> set[int]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        raise NotImplementedError
