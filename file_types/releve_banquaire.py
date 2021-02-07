import os
import cv2
import pandas as pd
from utils.deskew_image import deskew_img
from utils.process_table import process_tables
from utils.utils import get_json_from_file, pdf_to_jpg, save_cv_image
from utils.process_fields import get_agency_information, get_bank_id, get_client_information, get_date
from file_types.file_type import FileType


#TODO Améliorer les résultats


class ReleveBanquaire(FileType):
    
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
    
    def processing(self):
        extension = self.file_path.split('.')[-1]
        if extension == 'pdf':
            paths = pdf_to_jpg(self.file_path, self.folder_path)
        elif extension in ['jpg', 'jpeg']:
            paths = [self.file_path]
        else:
            print('Error: {} is not a valid PDF or JPG file'.format(self.file_path))
            return False

        for path in paths:

            # Convert tiff to cv2 img
            img = cv2.imread(path, 0)
            
            if img is None:
                print('Error while trying to load {}'.format(path))
                continue

            # Rotate img
            img = deskew_img(img)

            # Save image to jpg and remove tiff
            self.processed_file_path.append(save_cv_image(img, path, 'jpg', del_original=True))
            
        if len(paths) == 0:
            print('Error: no pages found in {}'.format(self.file_path))
            return False
        return True

    def parse_fields(self):
        
        debug_folder = os.path.join(self.folder_path, 'debug')
        if not os.path.exists(debug_folder):
            os.makedirs(debug_folder)
        
        # Load first page as 1D array
        first_page = cv2.imread(self.processed_file_path[0], 0)
        if first_page is None:
            print("Error : Can't load {}".format(self.processed_file_path[0]))
            return False
            
        #* Process bank id
        # Bank id should be on first page
        print('Finding bank...\r', end='')
        bank_id = get_bank_id(first_page)
        if bank_id is None:
            print('Error : unknown bank document.')
            return False
        # With bank id we can get bank information
        self.bank_utils = get_json_from_file('bank_configs/{}.json'.format(bank_id))
        self.dicts = get_json_from_file('dict.json')
        self.information["Bank name"] = self.bank_utils['name']
        print('Finding bank... DONE')
        
        #* Process fields
        print('Processing fields...\r', end='')
        # Process client information (should be in first page)
        self.information["Client full name"],\
        self.information["Client address"] = get_client_information(
            first_page,
            self.bank_utils,
            self.dicts,
            os.path.join(debug_folder, 'client_info.jpg') if self.debug else None
        )
        # Process Bank information
        self.information["Agency email"],\
        self.information["Agency phone"],\
        self.information["Agency address"] = get_agency_information(
            first_page,
            self.bank_utils,
            self.dicts,
            os.path.join(debug_folder, 'agency_info.jpg') if self.debug else None
        )
        self.information["Date"] = get_date(
            first_page,
            self.bank_utils,
            os.path.join(debug_folder, 'date_info.jpg') if self.debug else None
        )
        self.information["Date"] = self.information["Date"].strftime("%d %B %Y")
        
        infos_df = pd.DataFrame.from_dict(self.information, orient='index')
        infos_df.to_excel(self.excel_writer,
                          sheet_name=self.sheet_name,
                          startcol=0, startrow=self.row)
        self.row += len(self.information) + 2
        print('Processing fields... [DONE]')
        
        #* Process tables
        print('Processing tables...\r', end='')
        page_tables = []
        for i, path in enumerate(self.processed_file_path):
            page_tables += process_tables(
                path,
                arrange_mode=1,
                debug_folder=os.path.join(debug_folder, 'page_{}'.format(i)) if self.debug else None,
            )
            
        dfs_len = set([len(df.columns) for df in page_tables])
        self.statement_tables = [None] * len(dfs_len)
        for i, df_len in enumerate(dfs_len):
            for df in page_tables:
                if len(df.columns) == df_len:
                    if self.statement_tables[i] is not None:
                        df.columns = self.statement_tables[i].columns
                    self.statement_tables[i] = df if self.statement_tables[i] is None\
                        else pd.concat([self.statement_tables[i], df], ignore_index=True)

        self.statement_tables.sort(key = lambda df: len(df.index), reverse=True)
        tables_status = self.check_solde()
        # Save tables to excel files
        for i, df in enumerate(self.statement_tables):
            status_df = pd.DataFrame.from_dict(tables_status[i], orient='index', columns=['Description'])
            status_df.to_excel(self.excel_writer, sheet_name=self.sheet_name, startcol=0, startrow=self.row)
            self.row += 2
            df.to_excel(self.excel_writer, sheet_name=self.sheet_name, startcol=0, startrow=self.row)
            self.row += len(df.index) + 2
        print('Processing tables... [DONE]')
        return True
    
    @staticmethod
    def valid_value(val, dates):
        if val == '':
            return False
        return len([d for d in dates if d != '' and 'solde' not in d.lower()]) > 0
    
    def check_solde(self):
        cred_regx = '|'.join(self.dicts['credit'])
        deb_regx = '|'.join(self.dicts['debit'])
        date_regx = 'date'
        solde_regx = 'solde'
        status = [None] * len(self.statement_tables)
        for i, table in enumerate(self.statement_tables):
            table.columns = table.columns.str.strip().str.lower()
            #* Get credit and debit column names
            cred_col = table.filter(regex=cred_regx)
            deb_col = table.filter(regex=deb_regx)
            date_col = table.filter(regex=date_regx)
            
            col_empty = [cred_col.empty, deb_col.empty, date_col.empty]
            if any(col_empty):
                col_names = ['credit', 'debit', 'date']
                missing_col = [col for j, col in enumerate(col_names) if col_empty[j]]
                status[i] = { 'Unknown': 'columns missing : {}'.format(', '.join(missing_col)) }
                continue
            cred_col_name = list(cred_col.columns)
            deb_col_name = list(deb_col.columns)
            date_col_name = list(date_col.columns)
                    
            #* Get solde values
            other_soldes = [table[other_name].str.contains(solde_regx, case=False, na=False) for other_name in table.columns]
            other_solde = other_soldes[0]
            for j in range(1, len(other_soldes)):
                other_solde |= other_soldes[j]
            soldes = table[other_solde.values]
            solde = dict()
            solde['credit'] = [(val.replace(',', '.'), idx)\
                for val, idx in zip(soldes[cred_col_name[0]], soldes[cred_col_name[0]].index) if val != '']
            solde['debit'] = [(val.replace(',', '.'), idx)\
                for val, idx in zip(soldes[deb_col_name[0]], soldes[deb_col_name[0]].index) if val != '']
            
            if len(solde['credit']) < 2 and len(solde['debit']) < 2:
                status[i] = { 'Unknown': 'Not enough solde infos' }
                continue
            
            check_col_n = 'debit' if len(solde['debit']) >= 2 else 'credit'
            first_val, last_val = solde[check_col_n][0], solde[check_col_n][-1]
            sub_table = table[first_val[1]+1:last_val[1]]
            #* Get credit and debit values
            cred_names = cred_col_name + date_col_name
            cred_values = [float(row[0].replace(',', '.'))\
                for row in sub_table[cred_names].to_numpy() if self.valid_value(row[0], row[1:])]
            cred_val = sum(cred_values)
            deb_names = deb_col_name + date_col_name
            deb_values = [float(row[0].replace(',', '.'))\
                for row in sub_table[deb_names].to_numpy() if self.valid_value(row[0], row[1:])]
            deb_val = sum(deb_values)
            #* Calc solde final value with table values
            res = round(float(first_val[0]) + (deb_val - cred_val if check_col_n == 'debit' else cred_val - deb_val), 2)
            
            #* Set status depending on calculated solde value matching real final solde value
            if res == float(last_val[0]):
                status[i] = { 'Success': 'Table values match final solde value.' }
            else:
                status[i] = { 'Error' : "table values {} don't match final solde value {}.".format(res, float(last_val[0])) }
        return status