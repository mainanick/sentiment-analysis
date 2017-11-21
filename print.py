import sys

for progress in range(1000):
    sys.stdout.write("Progress {}   \r".format(progress))
    sys.stdout.flush()

