# SPDX-License-Identifier: Apache-2.0

import os

import aiohttp
from quart import Quart, make_response, request

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

app = Quart(__name__)


async def forward_request(url, data):
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
<<<<<<< HEAD
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"
        }
        async with session.post(url=url, json=data,
                                headers=headers) as response:
            if response.status == 200:
                # if response.headers.get('Transfer-Encoding') == 'chunked':
                if True:
                    async for chunk_bytes in response.content.iter_chunked(
                            1024):
=======
        headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}
        async with session.post(url=url, json=data, headers=headers) as response:
            if response.status == 200:
                # if response.headers.get('Transfer-Encoding') == 'chunked':
                if True:
                    async for chunk_bytes in response.content.iter_chunked(1024):
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
                        yield chunk_bytes
                else:
                    content = await response.read()
                    yield content


<<<<<<< HEAD
@app.route('/v1/completions', methods=['POST'])
=======
@app.route("/v1/completions", methods=["POST"])
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
async def handle_request():
    try:
        original_request_data = await request.get_json()

        prefill_request = original_request_data.copy()
        # change max_tokens = 1 to let it only do prefill
<<<<<<< HEAD
        prefill_request['max_tokens'] = 1

        # finish prefill
        async for _ in forward_request('http://localhost:8100/v1/completions',
                                       prefill_request):
            continue

        # return decode
        generator = forward_request('http://localhost:8200/v1/completions',
                                    original_request_data)
=======
        prefill_request["max_tokens"] = 1

        # finish prefill
        async for _ in forward_request(
            "http://localhost:8100/v1/completions", prefill_request
        ):
            continue

        # return decode
        generator = forward_request(
            "http://localhost:8200/v1/completions", original_request_data
        )
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        response = await make_response(generator)
        response.timeout = None

        return response

    except Exception as e:
        import sys
        import traceback
<<<<<<< HEAD
=======

>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        exc_info = sys.exc_info()
        print("Error occurred in disagg prefill proxy server")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))


<<<<<<< HEAD
if __name__ == '__main__':
=======
if __name__ == "__main__":
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    app.run(port=8000)
