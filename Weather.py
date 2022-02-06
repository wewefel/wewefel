import requests

def weather(city):
    API_KEY = "e1be7859ad5e2b65e7ac0e4b885f76dc"
    BASE_URL = "http://api.openweathermap.org/data/2.5/weather"

    request_url = f"{BASE_URL}?appid={API_KEY}&q={city}"
    response = requests.get(request_url)

    if response.status_code == 200:
        data = response.json()
        weather = data["weather"][0]["description"]
        print("Weather: ", weather)
        temperature = str(round((data["main"]["temp"] - 273.15) * (9/5) + 32))
        print("Temperature: ", temperature, "fahrenheit")
        chatbot_message = weather + temperature + "degrees fahrenheit"
        return chatbot_message
    else:
        print("An error occurred.")
        return "error"
