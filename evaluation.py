import json

# converting json file to list of dictionaries
# load json file
with open("dokument.json", "r", encoding="utf-8") as file:
    doc = json.load(file)

plaintiff_list = []
defendant_list = []

# iterate through list
for d in doc:

    # create empty dicts
    plaintiff_dict = dict.fromkeys["Kläger-Vortrag", "Kläger-Passage"]
    defendant_dict = dict.fromkeys["Beklagter-Vortrag", "Beklagter-Passage"]
    
    # get relevant keys
    # kläger-vortrag - kläger-dokument passage
    argument_plaintiff = d['Kläger-Vortrag']
    document_passage_plaintiff = d['Kläger-Dokument Passage']

    # only existing arguments and corresponding passages
    if argument_plaintiff != ' - ' and document_passage_plaintiff != ' - ':
        print(f"Klägervortrag: {argument_plaintiff} : Passage: {document_passage_plaintiff}")

        # assign filtered values to empty dict
        plaintiff_dict["Kläger-Vortrag"] = argument_plaintiff
        plaintiff_dict["Kläger-Passage"] = document_passage_plaintiff

        plaintiff_list.append(plaintiff_dict)


    # beklagter-vortrag - beklagter-dokument passage
    argument_defendant = d['Beklagter-Vortrag']
    document_passage_defendant = d['Beklagter-Dokument Passage']

    if argument_defendant != ' - ' and document_passage_defendant != ' - ':
        print(f"Beklagtervortrag: {argument_defendant} : Passage {document_passage_defendant}")

        # assign filtered values to empty dict
        defendant_dict["Kläger-Vortrag"] = argument_defendant
        defendant_dict["Kläger-Passage"] = document_passage_defendant

        defendant_list.append(defendant_dict)

    

    





    