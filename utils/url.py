import sys

import requests


def download_with_progressbar(url: str, file_name: str):
    with open(file_name, "wb") as f:
        print(f"Downloading {file_name}")
        response = requests.get(url, stream=True)
        total_length = response.headers.get("content-length")
        total_text_len = 50

        if total_length is None:  # no content length header
            f.write(response.content)
        else:
            dl = 0
            total_length = int(total_length)
            for data in response.iter_content(chunk_size=4096):
                dl += len(data)
                f.write(data)
                done = int(total_text_len * dl / total_length)
                sys.stdout.write(
                    f"\r[{'▆' * done}{' ' * (total_text_len - done)}]"
                    f"[{dl // 2 ** 20}MB/{total_length // 2 ** 20}MB]"
                )
                sys.stdout.flush()
