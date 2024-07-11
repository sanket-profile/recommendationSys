import sys

def getErrorDetails(message) -> str:
    _,_,exc_tb = sys.exc_info()
    if exc_tb:
        fileName = exc_tb.tb_frame.f_code.co_filename
        lineNumber = exc_tb.tb_lineno
        error_message = f"The error has occured in file - {fileName} on line number - {lineNumber}. ERROR IS - {message}"
    else:
        error_message = message

    return error_message

class CustomException(Exception):
    def __init__(self,message):
        super().__init__(message)
        self.error_message = getErrorDetails(message)

    def __str__(self):
        return self.error_message
    

if __name__ == "__main__":
    try:
        1/0
    except Exception as e:
        raise CustomException("Zero division ho gaya bhai")