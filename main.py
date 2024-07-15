import requests
import json

SNOWSTORM_API = "http://localhost:8080/"


def get_version():
    response = requests.get(SNOWSTORM_API + "version",
                            headers={"Accept": "application/json"})
    return response.json()


if __name__ == "__main__":
    print(json.dumps(get_version(), indent=4))
