import json
# Read file
with open("llm_generated_response.txt", "r", encoding="utf-8") as file:
    lines = file.readlines()

# cleanup
# skipping Documentheader
lines = lines[4::]

# empty list of dict
list_dict = []

for line in lines:

    # empty dict
    argument = dict.fromkeys(["Nummer", "Sachverhaltselement", "Kläger-Vortrag", "Kläger-Dokument Passage", "Beklagter-Vortrag", "Beklagter-Dokument Passage"])

    argument["Nummer"] = int(line.split('|')[1])
    argument["Sachverhaltselement"] = line.split('|')[2]
    argument["Kläger-Vortrag"] = line.split('|')[3]
    argument["Kläger-Dokument Passage"] = line.split('|')[4]
    argument["Beklagter-Vortrag"] = line.split('|')[5]
    argument["Beklagter-Dokument Passage"] = line.split('|')[6]

    list_dict.append(argument)

print(list_dict)

with open("dokument.json", "w", encoding="utf-8") as file:
    json.dump(list_dict, file, ensure_ascii=False, indent=2)




    





