from abc import ABC, abstractmethod


class FileType(ABC):
    """This class discribe a file class architecture
    """

    def __init__(self, file_path, doc_type, language, excel_writer, idx=0, debug=False):
        self.file_path = file_path
        self.language = language
        self.excel_writer = excel_writer
        self.debug = debug
        
        # Create folder where we will store our information and debug
        self.folder_path = '.'.join(self.file_path.split('.')[:-1])
        # Create sheet name where we will write this document information
        self.sheet_name = doc_type + ' {}'.format(idx)
        # Excel writing row
        self.row = 0
        # Paths of our porcessed files
        self.processed_file_path = []

    @abstractmethod
    def processing(self) -> bool:
        """This method is supposed to process the file to help ocr

        Returns:
            bool: True if no error occured, else False
        """
        raise NotImplementedError

    @abstractmethod
    def parse_fields(self) -> bool:
        """This method is supposed to parse file relevant fields

        Returns:
            bool: True if no error occured, else False
        """
        raise NotImplementedError
