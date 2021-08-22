import re
import sys

filename = sys.argv[1]
file = open(filename, "r")
output_file = open(filename + "_transf", "w")

should_process = True

last_value = None
last_time = None

while True:
    line = file.readline()

    if (len(line) == 0):
        break

    # print(line.startswith("--t"), should_process)
    if (line.startswith("[INFO] No visualization selected.")):
        should_process = True
        output_file.write(line)
    elif (line.startswith("Score")):
        should_process = False
        output_file.write(line)
        last_time = None
        last_value = None
    elif (line.startswith("--t") and should_process):
        parts = line.split(" ")
        current_value = round(float(parts[3]), 2)
        current_time = int(parts[1])

        if last_value is None:
            output_file.write("Obj " + line)
        else:
            output_file.write("Obj --t " + str(current_time) + " --o " + str(round(current_value - last_value, 2)) + "\n")
        
        last_value = current_value
        last_time = current_time
    else:
        output_file.write(line)