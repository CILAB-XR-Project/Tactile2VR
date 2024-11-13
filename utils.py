import sys
from const import ACTIVITY_LIST

class DualOutput:
    def __init__(self, filepath):
        self.file = open(filepath, 'w')  # Open the file where you want to store the outputs
        self.console = sys.stdout  # Save a reference to the original standard output

    def write(self, message):
        self.console.write(message)  # Write to the console
        self.file.write(message)  # Write to the file

    def flush(self):  # Needed for compatibility with sys.stdout
        self.console.flush()
        self.file.flush()

    def __enter__(self):
        sys.stdout = self  # Redirect standard output to this DualOutput object
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.console  # Restore standard output to the console
        self.file.close()  # Close the file


def get_action_type(string):
    action = None
    for activity_can in ACTIVITY_LIST:
        if activity_can in string:
            action = activity_can
    if action == "Walking":
        assert ("SideWalking" not in string) and ("BackwardWalking" not in string) and ("InPlaceWalking" not in string)
    return action