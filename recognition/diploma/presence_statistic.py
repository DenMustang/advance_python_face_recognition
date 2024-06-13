import matplotlib.pyplot as plt
from datetime import datetime

presence_data = []
with open("presence.txt", "r") as file:
    for line in file:
        line = line.strip()
        date_str, student_name = line.split(" - ")
        date = datetime.strptime(date_str, "%d/%m/%Y")
        presence_data.append((date, student_name))

presence_data.sort(key=lambda x: x[0])

dates = [data[0] for data in presence_data]
students = [data[1] for data in presence_data]

student_counts = {}
for student in students:
    student_counts[student] = student_counts.get(student, 0) + 1

x_values = list(student_counts.keys())
y_values = list(student_counts.values())

plt.bar(x_values, y_values)
plt.xlabel("Students")
plt.ylabel("Number of Occurrences")
plt.title("Presence of Students Over Time")
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()
