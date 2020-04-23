import json
import random
import requests as rq

if __name__ == '__main__':
    vectors = [[random.random() for _ in range(128)] for _ in range(10000)]
    request = {
        "vectors": vectors
    }

    data = json.dumps(request)
    headers = {"Content-Type": "application/json"}
    response = rq.post("http://127.0.0.1:19121/", data=data, headers=headers)
