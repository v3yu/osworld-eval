def clean_url(url: str) -> str:
    url = str(url)
    if url.endswith("/"):
        url = url[:-1]
    return url

def clean_answer(answer: str) -> str:
    answer = answer.strip("'").strip('"')
    answer = answer.lower()
    return answer
