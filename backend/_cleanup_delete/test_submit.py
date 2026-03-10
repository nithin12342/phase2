import requests
data = {"survey_data": '{"gender":"Male","country":"India"}'}
files = {"photo": ("dummy.jpg", b"\xFF\xD8\xFF\xE0", "image/jpeg")}
resp = requests.post("http://localhost:8000/api/v1/submit-survey", data=data, files=files)
print(resp.json())
