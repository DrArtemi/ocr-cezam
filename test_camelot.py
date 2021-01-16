import camelot

pdf = "/media/adrien/Shared/Work/Cezam/documents/OCR/Relevés bancaires/CIC_C01_Relevé_Juillet_2018.pdf"
# pdf = "/media/adrien/Shared/Work/Cezam/documents/OCR/pdf_examples/compte joint_BANQUE POP_012017_bis.pdf"

tables = camelot.read_pdf(pdf, pages='1-end', flavor="lattice")

print(tables)

for table in tables:
    print(table.df)
    
    # pd.concat([Series(row['var2'], row['var1'].split(','))              
    #                 for _, row in a.iterrows()]).reset_index()