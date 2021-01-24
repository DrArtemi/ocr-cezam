# ocr-cezam

Extract information from PDF to Excel file

## Dependencies

`conda env create -f environment.yml`

## Config file

{
    name": "Excel name",
    "account_statements": [
        "/absolute/path/to/file_1.pdf",
        "/absolute/path/to/file_2.pdf",
    ],
    "tax_notices": [
        "/absolute/path/to/file_3.pdf",
    ],
    "identity_documents": [
        "/absolute/path/to/file_4.pdf",
    ]
}

## Command line

### Example

`python ocr_cezam.py --config config.json --lang fra`