from file_types.file_type import FileType


class Bilan(FileType):
    
    def __init__(self, file_path, language, excel_writer, idx=0, debug=False):
        super().__init__(file_path, language, excel_writer, idx=idx, debug=debug)