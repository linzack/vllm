# SPDX-License-Identifier: Apache-2.0

import requests
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (BatchSpanProcessor,
                                            ConsoleSpanExporter)
from opentelemetry.trace import SpanKind, set_tracer_provider
from opentelemetry.trace.propagation.tracecontext import (
    TraceContextTextMapPropagator)

trace_provider = TracerProvider()
set_tracer_provider(trace_provider)

trace_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
trace_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

tracer = trace_provider.get_tracer("dummy-client")

url = "http://localhost:8000/v1/completions"
with tracer.start_as_current_span("client-span", kind=SpanKind.CLIENT) as span:
    prompt = "San Francisco is a"
    span.set_attribute("prompt", prompt)
    headers = {}
    TraceContextTextMapPropagator().inject(headers)
    payload = {
        "model": "facebook/opt-125m",
        "prompt": prompt,
        "max_tokens": 10,
<<<<<<< HEAD
        "best_of": 20,
=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        "n": 3,
        "use_beam_search": "true",
        "temperature": 0.0,
        # "stream": True,
    }
    response = requests.post(url, headers=headers, json=payload)
