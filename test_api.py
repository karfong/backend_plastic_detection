import requests
import json

API_URL = "http://127.0.0.1:5000/detect"
image_path = "images/pet1.jpg"

with open(image_path, "rb") as img_file:
    files = {"image": img_file}
    response = requests.post(API_URL, files=files)

# Pretty-print the JSON response
if response.status_code == 200:
    formatted_response = json.dumps(response.json(), indent=4, sort_keys=True)
    print("\n🔍 Detection Results:\n")
    print(formatted_response)
else:
    print("\n❌ Error:", response.status_code, response.text)
