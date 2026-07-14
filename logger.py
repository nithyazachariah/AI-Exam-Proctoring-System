from datetime import datetime
import csv
import os

LOG_FILE = "logs/events.csv"

def log_event(event):

    file_exists = os.path.isfile(LOG_FILE)

    with open(LOG_FILE, "a", newline="") as file:

        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(["Timestamp", "Event"])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            event
        ])