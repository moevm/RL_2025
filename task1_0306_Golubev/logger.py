class Logger:
    def __init__(self, path_to_file):
        self.path_to_file = path_to_file
    

    def info(self, message):
        print(message)
        with open(self.path_to_file, 'a') as file:
            file.write(message + '\n')