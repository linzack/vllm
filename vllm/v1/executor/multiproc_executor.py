# SPDX-License-Identifier: Apache-2.0
<<<<<<< HEAD

=======
import multiprocessing
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
import os
import pickle
import signal
import sys
<<<<<<< HEAD
import time
import weakref
from dataclasses import dataclass
from enum import Enum, auto
from functools import partial
from multiprocessing.process import BaseProcess
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cloudpickle
import psutil
import zmq
=======
import threading
import time
import traceback
import weakref
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum, auto
from functools import partial
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from threading import Thread
from typing import Any, Callable, Optional, Union, cast

import cloudpickle
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

from vllm.config import VllmConfig
from vllm.distributed import (destroy_distributed_environment,
                              destroy_model_parallel)
from vllm.distributed.device_communicators.shm_broadcast import (Handle,
                                                                 MessageQueue)
from vllm.executor.multiproc_worker_utils import (
    _add_prefix, set_multiprocessing_worker_envs)
from vllm.logger import init_logger
from vllm.utils import (get_distributed_init_method, get_mp_context,
<<<<<<< HEAD
                        get_open_port, get_open_zmq_ipc_path, zmq_socket_ctx)
from vllm.v1.executor.abstract import Executor
=======
                        get_open_port)
from vllm.v1.executor.abstract import Executor, FailureCallback
from vllm.v1.outputs import ModelRunnerOutput
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
from vllm.worker.worker_base import WorkerWrapperBase

logger = init_logger(__name__)

POLLING_TIMEOUT_MS = 5000
POLLING_TIMEOUT_S = POLLING_TIMEOUT_MS // 1000

<<<<<<< HEAD
=======
EXECUTE_MODEL_TIMEOUT_S = 40

>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

class MultiprocExecutor(Executor):

    def _init_executor(self) -> None:
        # Call self.shutdown at exit to clean up
        # and ensure workers will be terminated.
        self._finalizer = weakref.finalize(self, self.shutdown)
<<<<<<< HEAD

        # The child processes will send SIGUSR1 when unrecoverable
        # errors happen.
        def sigusr1_handler(signum, frame):
            logger.fatal(
                "MulitprocExecutor got fatal signal from worker processes, "
                "shutting down. See stack trace above for root cause issue.")
            # Propagate error up to parent process.
            parent_process = psutil.Process().parent()
            parent_process.send_signal(signal.SIGUSR1)
            self.shutdown()

        signal.signal(signal.SIGUSR1, sigusr1_handler)

        self.world_size = self.parallel_config.world_size
        tensor_parallel_size = self.parallel_config.tensor_parallel_size
        assert self.world_size == tensor_parallel_size, (
            f"world_size ({self.world_size}) must be equal to the "
            f"tensor_parallel_size ({tensor_parallel_size}). "
            f"Pipeline parallelism is not yet implemented in v1")
=======
        self.is_failed = False
        self.shutdown_event = threading.Event()
        self.failure_callback: Optional[FailureCallback] = None

        self.world_size = self.parallel_config.world_size
        tensor_parallel_size = self.parallel_config.tensor_parallel_size
        pp_parallel_size = self.parallel_config.pipeline_parallel_size
        assert self.world_size == tensor_parallel_size * pp_parallel_size, (
            f"world_size ({self.world_size}) must be equal to the "
            f"tensor_parallel_size ({tensor_parallel_size}) x pipeline"
            f"_parallel_size ({pp_parallel_size}). ")
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

        # Set multiprocessing envs that are common to V0 and V1
        set_multiprocessing_worker_envs(self.parallel_config)

        # Multiprocessing-based executor does not support multi-node setting.
        # Since it only works for single node, we can use the loopback address
        # 127.0.0.1 for communication.
        distributed_init_method = get_distributed_init_method(
            "127.0.0.1", get_open_port())

        # Initialize worker and set up message queues for SchedulerOutputs
        # and ModelRunnerOutputs
        self.rpc_broadcast_mq = MessageQueue(self.world_size, self.world_size)
        scheduler_output_handle = self.rpc_broadcast_mq.export_handle()

        # Create workers
<<<<<<< HEAD
        self.workers: List[WorkerProcHandle] = []
        for rank in range(self.world_size):
            worker = WorkerProc.make_worker_process(self.vllm_config, rank,
                                                    rank,
                                                    distributed_init_method,
                                                    scheduler_output_handle)
            self.workers.append(worker)

        # Ensure message queues are ready. Will deadlock if re-ordered
        # Must be kept consistent with the WorkerProc
        self.rpc_broadcast_mq.wait_until_ready()
        for w in self.workers:
            w.worker_response_mq.wait_until_ready()
=======
        unready_workers: list[UnreadyWorkerProcHandle] = []
        success = False
        try:
            for rank in range(self.world_size):
                unready_workers.append(
                    WorkerProc.make_worker_process(
                        vllm_config=self.vllm_config,
                        local_rank=rank,
                        rank=rank,
                        distributed_init_method=distributed_init_method,
                        input_shm_handle=scheduler_output_handle,
                    ))

            # Workers must be created before wait_for_ready to avoid
            # deadlock, since worker.init_device() does a device sync.
            self.workers = WorkerProc.wait_for_ready(unready_workers)

            # Ensure message queues are ready. Will deadlock if re-ordered
            # Must be kept consistent with the WorkerProc.
            self.rpc_broadcast_mq.wait_until_ready()
            for w in self.workers:
                w.worker_response_mq.wait_until_ready()

            self.start_worker_monitor()
            success = True
        finally:
            if not success:
                # Clean up the worker procs if there was a failure.
                self._ensure_worker_termination(
                    [w.proc for w in unready_workers])

        # For pipeline parallel, we use a thread pool for asynchronous
        # execute_model.
        self.io_thread_pool: Optional[ThreadPoolExecutor] = None
        if self.max_concurrent_batches > 1:
            # Note: must use only 1 IO thread to keep dequeue sequence
            # from the response queue
            self.io_thread_pool = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="mp_exec_io")

        self.output_rank = self._get_output_rank()

    def start_worker_monitor(self):
        workers = self.workers
        self_ref = weakref.ref(self)

        # Monitors worker process liveness. If any die unexpectedly,
        # logs an error, shuts down the executor and invokes the failure
        # callback to inform the engine.
        def monitor_workers():
            sentinels = [h.proc.sentinel for h in workers]
            died = multiprocessing.connection.wait(sentinels)
            _self = self_ref()
            if not _self or getattr(_self, 'shutting_down', False):
                return
            _self.is_failed = True
            proc_name = next(h.proc.name for h in workers
                             if h.proc.sentinel == died[0])
            logger.error(
                "Worker proc %s died unexpectedly, "
                "shutting down executor.", proc_name)
            _self.shutdown()
            callback = _self.failure_callback
            if callback is not None:
                _self.failure_callback = None
                callback()

        Thread(target=monitor_workers,
               daemon=True,
               name="MultiprocWorkerMonitor").start()

    def register_failure_callback(self, callback: FailureCallback):
        if self.is_failed:
            callback()
        else:
            self.failure_callback = callback

    def execute_model(
        self,
        scheduler_output,
    ) -> Union[ModelRunnerOutput, Future[ModelRunnerOutput]]:
        (output, ) = self.collective_rpc("execute_model",
                                         args=(scheduler_output, ),
                                         unique_reply_rank=self.output_rank,
                                         non_block=self.max_concurrent_batches
                                         > 1,
                                         timeout=EXECUTE_MODEL_TIMEOUT_S)
        return output
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

    def collective_rpc(self,
                       method: Union[str, Callable],
                       timeout: Optional[float] = None,
<<<<<<< HEAD
                       args: Tuple = (),
                       kwargs: Optional[Dict] = None) -> List[Any]:
        start_time = time.monotonic()
=======
                       args: tuple = (),
                       kwargs: Optional[dict] = None,
                       non_block: bool = False,
                       unique_reply_rank: Optional[int] = None) -> list[Any]:
        if self.is_failed:
            raise RuntimeError("Executor failed.")

        deadline = None if timeout is None else time.monotonic() + timeout
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        kwargs = kwargs or {}

        # NOTE: If the args are heterogeneous, then we pack them into a list,
        # and unpack them in the method of every worker, because every worker
        # knows their own rank.
        try:
            if isinstance(method, str):
                send_method = method
            else:
                send_method = cloudpickle.dumps(
                    method, protocol=pickle.HIGHEST_PROTOCOL)
<<<<<<< HEAD
            self.rpc_broadcast_mq.enqueue((send_method, args, kwargs))

            responses = [None] * self.world_size
            for w in self.workers:
                dequeue_timeout = timeout - (time.monotonic() - start_time
                                             ) if timeout is not None else None
                status, result = w.worker_response_mq.dequeue(
                    timeout=dequeue_timeout)

                if status != WorkerProc.ResponseStatus.SUCCESS:
                    if isinstance(result, Exception):
                        raise result
                    else:
                        raise RuntimeError("Worker failed")

                responses[w.rank] = result
=======
            self.rpc_broadcast_mq.enqueue(
                (send_method, args, kwargs, unique_reply_rank))

            workers = (self.workers[unique_reply_rank],
                       ) if unique_reply_rank is not None else self.workers
            responses = []

            def get_response(w: WorkerProcHandle,
                             dequeue_timeout: Optional[float] = None,
                             cancel_event: Optional[threading.Event] = None):
                status, result = w.worker_response_mq.dequeue(
                    timeout=dequeue_timeout, cancel=cancel_event)

                if status != WorkerProc.ResponseStatus.SUCCESS:
                    raise RuntimeError(
                        f"Worker failed with error '{result}', please check the"
                        " stack trace above for the root cause")
                return result

            for w in workers:
                dequeue_timeout = None if deadline is None else (
                    deadline - time.monotonic())

                if non_block:
                    result = self.io_thread_pool.submit(  # type: ignore
                        get_response, w, dequeue_timeout, self.shutdown_event)
                else:
                    result = get_response(w, dequeue_timeout)

                responses.append(result)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

            return responses
        except TimeoutError as e:
            raise TimeoutError(f"RPC call to {method} timed out.") from e
<<<<<<< HEAD
        except Exception as e:
            # Re-raise any other exceptions
            raise e

    def _ensure_worker_termination(self):
=======

    @staticmethod
    def _ensure_worker_termination(worker_procs: list[BaseProcess]):
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        """Ensure that all worker processes are terminated. Assumes workers have
        received termination requests. Waits for processing, then sends
        termination and kill signals if needed."""

        def wait_for_termination(procs, timeout):
            if not time:
                # If we are in late stage shutdown, the interpreter may replace
                # `time` with `None`.
                return all(not proc.is_alive() for proc in procs)
            start_time = time.time()
            while time.time() - start_time < timeout:
                if all(not proc.is_alive() for proc in procs):
                    return True
                time.sleep(0.1)
            return False

        # Send SIGTERM if still running
<<<<<<< HEAD
        active_procs = [w.proc for w in self.workers if w.proc.is_alive()]
=======
        active_procs = [proc for proc in worker_procs if proc.is_alive()]
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        for p in active_procs:
            p.terminate()
        if not wait_for_termination(active_procs, 4):
            # Send SIGKILL if still running
            active_procs = [p for p in active_procs if p.is_alive()]
            for p in active_procs:
                p.kill()

<<<<<<< HEAD
        self._cleanup_sockets()

    def _cleanup_sockets(self):
        for w in self.workers:
            # Remove the zmq ipc socket file
            socket_path = w.ready_path.replace("ipc://", "")
            if os and os.path.exists(socket_path):
                os.remove(socket_path)

    def shutdown(self):
        """Properly shut down the executor and its workers"""
        if getattr(self, 'shutting_down', False):
            self.shutting_down = True
            for w in self.workers:
                w.worker_response_mq = None
            self._ensure_worker_termination()
=======
    def shutdown(self):
        """Properly shut down the executor and its workers"""
        if not getattr(self, 'shutting_down', False):
            self.shutting_down = True
            self.shutdown_event.set()

            if self.io_thread_pool is not None:
                self.io_thread_pool.shutdown(wait=False, cancel_futures=True)
                self.io_thread_pool = None

            if workers := getattr(self, 'workers', None):
                for w in workers:
                    w.worker_response_mq = None
                self._ensure_worker_termination([w.proc for w in workers])
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

        self.rpc_broadcast_mq = None

    def check_health(self) -> None:
        self.collective_rpc("check_health", timeout=10)
        return

<<<<<<< HEAD
=======
    @property
    def max_concurrent_batches(self) -> int:
        return self.parallel_config.pipeline_parallel_size

    def _get_output_rank(self) -> int:
        # Only returns ModelRunnerOutput from TP rank=0 and PP rank=-1
        # (the first TP worker of the last PP stage).
        # Example:
        # Assuming TP=8, PP=4, then the world_size=32
        # 0-7, PP rank 0
        # 8-15, PP rank 1
        # 16-23, PP rank 2
        # 24-31, PP rank 3
        # so world_size - tp_size = 32 - 8 = 24 should be PP rank = -1 (i.e. 3)
        return self.world_size - self.parallel_config.tensor_parallel_size


@dataclass
class UnreadyWorkerProcHandle:
    """WorkerProcess handle before READY."""
    proc: BaseProcess
    rank: int
    ready_pipe: Connection

>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

@dataclass
class WorkerProcHandle:
    proc: BaseProcess
    rank: int
<<<<<<< HEAD
    ready_path: str
    worker_response_mq: MessageQueue  # The worker process writes to this MQ

=======
    worker_response_mq: MessageQueue  # The worker process writes to this MQ

    @classmethod
    def from_unready_handle(
            cls, unready_handle: UnreadyWorkerProcHandle,
            worker_response_mq: MessageQueue) -> "WorkerProcHandle":
        return cls(
            proc=unready_handle.proc,
            rank=unready_handle.rank,
            worker_response_mq=worker_response_mq,
        )

>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

class WorkerProc:
    """Wrapper that runs one Worker in a separate process."""

    READY_STR = "READY"

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        input_shm_handle: Handle,
<<<<<<< HEAD
        ready_path: str,
=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    ):
        self.rank = rank
        wrapper = WorkerWrapperBase(vllm_config=vllm_config, rpc_rank=rank)
        # TODO: move `init_worker` to executor level as a collective rpc call
<<<<<<< HEAD
        all_kwargs: List[Dict] = [
            {} for _ in range(vllm_config.parallel_config.world_size)
        ]
=======
        all_kwargs: list[dict] = [
            {} for _ in range(vllm_config.parallel_config.world_size)
        ]
        is_driver_worker = (
            rank % vllm_config.parallel_config.tensor_parallel_size == 0)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        all_kwargs[rank] = {
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
<<<<<<< HEAD
        }
        wrapper.init_worker(all_kwargs)
        self.worker = wrapper.worker
=======
            "is_driver_worker": is_driver_worker,
        }
        wrapper.init_worker(all_kwargs)
        self.worker = wrapper
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

        pid = os.getpid()
        _add_prefix(sys.stdout, f"VllmWorker rank={rank}", pid)
        _add_prefix(sys.stderr, f"VllmWorker rank={rank}", pid)

        # Initialize MessageQueue for receiving SchedulerOutput
        self.rpc_broadcast_mq = MessageQueue.create_from_handle(
            input_shm_handle, self.worker.rank)

        # Initializes a message queue for sending the model output
        self.worker_response_mq = MessageQueue(1, 1)
<<<<<<< HEAD
        worker_response_mq_handle = self.worker_response_mq.export_handle()

        # Send Readiness signal to EngineCore process.
        with zmq_socket_ctx(ready_path, zmq.constants.PUSH) as ready_socket:
            payload = pickle.dumps(worker_response_mq_handle,
                                   protocol=pickle.HIGHEST_PROTOCOL)
            ready_socket.send_string(WorkerProc.READY_STR)
            ready_socket.send(payload)

        wrapper.init_device()
=======

        # Initialize device and loads weights
        self.worker.init_device()
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        self.worker.load_model()

    @staticmethod
    def make_worker_process(
            vllm_config: VllmConfig,
            local_rank: int,
            rank: int,
            distributed_init_method: str,
            input_shm_handle,  # Receive SchedulerOutput
<<<<<<< HEAD
    ) -> WorkerProcHandle:
        context = get_mp_context()

        # ZMQ path for worker to send ready message and shm_broadcast handle
        # back to core process.
        ready_path = get_open_zmq_ipc_path()
=======
    ) -> UnreadyWorkerProcHandle:
        context = get_mp_context()
        # (reader, writer)
        reader, writer = context.Pipe(duplex=False)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

        process_kwargs = {
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
            "input_shm_handle": input_shm_handle,
<<<<<<< HEAD
            "ready_path": ready_path,
=======
            "ready_pipe": (reader, writer),
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        }
        # Run EngineCore busy loop in background process.
        proc = context.Process(target=WorkerProc.worker_main,
                               kwargs=process_kwargs,
<<<<<<< HEAD
                               daemon=True)
        proc.start()

        # Wait for startup
        worker_response_mq_handle = WorkerProc.wait_for_startup(
            proc, ready_path)

        worker_response_mq = MessageQueue.create_from_handle(
            worker_response_mq_handle, 0)

        return WorkerProcHandle(proc, rank, ready_path, worker_response_mq)
=======
                               name=f"VllmWorker-{rank}",
                               daemon=True)

        proc.start()
        writer.close()
        return UnreadyWorkerProcHandle(proc, rank, reader)

    @staticmethod
    def wait_for_ready(
        unready_proc_handles: list[UnreadyWorkerProcHandle]
    ) -> list[WorkerProcHandle]:

        e = Exception("WorkerProc initialization failed due to "
                      "an exception in a background process. "
                      "See stack trace for root cause.")

        pipes = {handle.ready_pipe: handle for handle in unready_proc_handles}
        ready_proc_handles: list[Optional[WorkerProcHandle]] = (
            [None] * len(unready_proc_handles))
        while pipes:
            ready = multiprocessing.connection.wait(pipes.keys())
            for pipe in ready:
                assert isinstance(pipe, Connection)
                try:
                    # Wait until the WorkerProc is ready.
                    unready_proc_handle = pipes.pop(pipe)
                    response: dict[str, Any] = pipe.recv()
                    if response["status"] != "READY":
                        raise e

                    # Extract the message queue handle.
                    worker_response_mq = MessageQueue.create_from_handle(
                        response["handle"], 0)
                    ready_proc_handles[unready_proc_handle.rank] = (
                        WorkerProcHandle.from_unready_handle(
                            unready_proc_handle, worker_response_mq))

                except EOFError:
                    e.__suppress_context__ = True
                    raise e from None

                finally:
                    # Close connection.
                    pipe.close()

        return cast(list[WorkerProcHandle], ready_proc_handles)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

    def shutdown(self):
        self.rpc_broadcast_mq = None
        self.worker_response_mq = None
        destroy_model_parallel()
        destroy_distributed_environment()

    @staticmethod
    def worker_main(*args, **kwargs):
        """ Worker initialization and execution loops.
        This runs a background process """

        # Signal handler used for graceful termination.
        # SystemExit exception is only raised once to allow this and worker
        # processes to terminate without error
        shutdown_requested = False

        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                raise SystemExit()

        # Either SIGTERM or SIGINT will terminate the worker
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        worker = None
<<<<<<< HEAD
        try:
            worker = WorkerProc(*args, **kwargs)

=======
        # tuple[Connection, Connection]
        reader, ready_writer = kwargs.pop("ready_pipe")
        try:
            reader.close()
            worker = WorkerProc(*args, **kwargs)

            # Send READY once we know everything is loaded
            ready_writer.send({
                "status":
                WorkerProc.READY_STR,
                "handle":
                worker.worker_response_mq.export_handle(),
            })

>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
            # Ensure message queues are ready. Will deadlock if re-ordered.
            # Must be kept consistent with the Executor
            worker.rpc_broadcast_mq.wait_until_ready()
            worker.worker_response_mq.wait_until_ready()
<<<<<<< HEAD

            worker.worker_busy_loop()

        except SystemExit:
            logger.debug("Worker interrupted.")

        except Exception:
            # worker_busy_loop sends exceptions exceptons to Executor
            # for shutdown, but if there is an error in startup or an
            # error with IPC itself, we need to alert the parent.
            psutil.Process().parent().send_signal(signal.SIGUSR1)
            raise

        finally:
            # Clean up once worker exits busy loop
            if worker is not None:
                worker.shutdown()
                worker = None

    @staticmethod
    def wait_for_startup(
        proc: BaseProcess,
        ready_path: str,
    ) -> Optional[Handle]:
        """Wait until the Worker is ready."""
        with zmq_socket_ctx(ready_path, zmq.constants.PULL) as socket:

            # Wait for Worker to send READY.
            while socket.poll(timeout=POLLING_TIMEOUT_MS) == 0:
                logger.debug("Waiting for WorkerProc to startup.")

                if not proc.is_alive():
                    raise RuntimeError("WorkerProc failed to start.")

            message = socket.recv_string()
            assert message == WorkerProc.READY_STR
            handle_frame = socket.recv(copy=False)
            handle = pickle.loads(handle_frame.buffer)
            return handle
=======
            ready_writer.close()
            ready_writer = None

            worker.worker_busy_loop()

        except Exception:
            # NOTE: if an Exception arises in busy_loop, we send
            # a FAILURE message over the MQ RPC to notify the Executor,
            # which triggers system shutdown.
            # TODO(rob): handle case where the MQ itself breaks.

            if ready_writer is not None:
                logger.exception("WorkerProc failed to start.")
            else:
                logger.exception("WorkerProc failed.")

            # The parent sends a SIGTERM to all worker processes if
            # any worker dies. Set this value so we don't re-throw
            # SystemExit() to avoid zmq exceptions in __del__.
            shutdown_requested = True

        finally:
            if ready_writer is not None:
                ready_writer.close()
            # Clean up once worker exits busy loop
            if worker is not None:
                worker.shutdown()
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

    class ResponseStatus(Enum):
        SUCCESS = auto()
        FAILURE = auto()

    def worker_busy_loop(self):
        """Main busy loop for Multiprocessing Workers"""
        while True:
<<<<<<< HEAD
            method, args, kwargs = self.rpc_broadcast_mq.dequeue()
=======
            method, args, kwargs, output_rank = self.rpc_broadcast_mq.dequeue()
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

            try:
                if isinstance(method, str):
                    func = getattr(self.worker, method)
                elif isinstance(method, bytes):
                    func = partial(cloudpickle.loads(method), self.worker)
                output = func(*args, **kwargs)
            except Exception as e:
<<<<<<< HEAD
                self.worker_response_mq.enqueue(
                    (WorkerProc.ResponseStatus.FAILURE, e))
                logger.exception("WorkerProc hit an exception: %s", exc_info=e)
                continue

            self.worker_response_mq.enqueue(
                (WorkerProc.ResponseStatus.SUCCESS, output))
=======
                # Notes have been introduced in python 3.11
                if hasattr(e, "add_note"):
                    e.add_note(traceback.format_exc())
                logger.exception("WorkerProc hit an exception.")
                # exception might not be serializable, so we convert it to
                # string, only for logging purpose.
                if output_rank is None or self.rank == output_rank:
                    self.worker_response_mq.enqueue(
                        (WorkerProc.ResponseStatus.FAILURE, str(e)))
                continue

            if output_rank is None or self.rank == output_rank:
                self.worker_response_mq.enqueue(
                    (WorkerProc.ResponseStatus.SUCCESS, output))
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
