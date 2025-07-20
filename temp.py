def get_cpu_temperature():
    with open("/sys/class/thermal/thermal_zone0/temp", "r") as file:
        temp = file.read().strip()
        return float(temp) / 1000  # Convert millidegrees to Celsius
import time
while True:
    if __name__ == "__main__":
        cpu_temp = get_cpu_temperature()
        print(f"CPU Temperature: {cpu_temp:.2f}°C")
    time.sleep(2)
