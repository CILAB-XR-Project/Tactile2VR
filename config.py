from const import ACTIVITY_LIST, VR_INDEXS
import json
# Base Configuration Class
# Don't use this class directly. Instead, sub-class it and override
# the configurations you need to change.

class BaseConfig(object):
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """
    NAME = None  # Override in sub-classes
    # NUMBER OF GPUs to use. For CPU use 0
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.0001

    # Sliding window size for tactile signal input
    WINDOW_SIZE = 20
    # whether predict middle frame or last frame of the sliding window
    PREDICT_MIDDLE = True

    # Shuffle dataloader
    SHUFFLE = False

    def to_dict(self):
        """Convert all configuration settings to a dictionary."""
        config_dict = {}
        for attribute_name in dir(self):
            # Ensure the attribute is not a built-in attribute
            if not attribute_name.startswith("__"):
                # Get the attribute value
                attribute_value = getattr(self, attribute_name)
                # Ensure the attribute is not a method or callable
                if not callable(attribute_value):
                    config_dict[attribute_name] = attribute_value
        return config_dict

    def save_to_json(self, file_path):
        """Save configuration settings to a JSON file."""
        config_dict = self.to_dict()
        with open(file_path, 'w') as file:
            json.dump(config_dict, file, indent=4)
        print(f"Configuration saved to {file_path}")

    def load_from_json(self, file_path):
        """Load configuration settings from a JSON file and update instance attributes."""
        with open(file_path, 'r') as file:
            config_dict = json.load(file)

        for key, value in config_dict.items():
            if not hasattr(self, key):
                print(f"Warning: {key} is not a valid configuration property")
            setattr(self, key, value)

    def print(self):
        print_square(self.to_dict())


def print_square(dictionary):
    for key in dictionary.keys():
        if "float" in str(type(dictionary[key])):
            newval = round(float(dictionary[key]), 4)
            dictionary[key] = newval

    front_lens = []
    back_lens = []
    for key in dictionary.keys():
        front_lens.append(len(key))
        back_lens.append(len(str(dictionary[key])))
    front_len = max(front_lens)
    back_len = max(back_lens)

    strings = []
    for key in dictionary.keys():
        string = "| {0:<{2}} | {1:<{3}} |".format(key, str(dictionary[key]), front_len, back_len)
        strings.append(string)

    max_len = max([len(i) for i in strings])

    text = ""
    text += "-"*max_len + "\n"
    for string in strings:
        text += string + "\n"
    text += "-"*max_len + "\n"
    print(text)
    return text


# ================================
# Configurations

class Tactile2PoseConfig(BaseConfig):
    NAME = "tactile2pose"
    WINDOW_SIZE = 20
    PREDICT_MIDDLE = False
    ACTION_LIST = ACTIVITY_LIST
    
    CACHE_SIZE = 125
    EPOCHS = 50
    BATCH_SIZE = 64
    ONLY_LOWER_BODY = False
    KP_NUM = 19


