from file_types.file_type import FileType


#TODO Améliorer les résultats

class DocumentIdentite(FileType):
    
    def __init__(self, file_path, language, excel_writer, idx=0, debug=False):
        super().__init__(file_path, language, excel_writer, idx=idx, debug=debug)
        
        # Bank infos
        self.information = {
            "Bank name": "N/A",
            "Agency address": "N/A",
            "Agency phone": "N/A",
            "Agency email": "N/A",
            "Consultant phone": "N/A",
            "Consultant email": "N/A",
            "Client full name": "N/A",
            "Client address": "N/A",
            "Date": "N/A"
        }