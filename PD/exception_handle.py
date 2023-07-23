
class OutOfBounds(Exception):

    def __init__(self, value):

        self.value = value

    def __str__(self, value):

        return f"value: {self.value} supplied is greater than 1 or less than 0"
