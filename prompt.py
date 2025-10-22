from langchain.prompts import PromptTemplate

def instruction_prompt(num_factual_elements: int) -> str:

    prompt_template = "#### Persona #### \n" \
    "{persona}" \
    "#### Aufgabenstellung #### \n" \
    "{instruction} \n\n" \
    "#### Aufgabenbeschreibung #### \n" \
    "{context} \n\n" \
    "#### Formatierung der Ausgabe #### \n" \
    "{format} \n\n" \
    "#### Definition der Zielgruppe #### \n" \
    "{audience} \n\n" \
    "#### Tonalität des Dokuments #### \n" \
    "{tone} \n\n" \
    "#### Regeln #### \n" \
    "{constraints} \n\n" \
    "#### Arbeitsschritte #### \n" \
    "{cot_instruction} \n\n" \
    "#### Anzahl der Sachverhaltselemente #### \n" \
    "In den vorliegenden Dokumenten liegen mindestens {num_factual_elements} Sachverhaltselemente vor. Diese Zahl an Elementen muss mindestens in der Relationstabelle vorkommen. \n\n" \
    "#### Ground Truth #### \n" \
    "{ground_truth_instruction}" 



    persona = "Du bist ein Richter an einem deutschen Gericht. Du wirst mit Streitigkeiten die sich innerhalb des deutschen Zivilrechts bewegen betraut."
    instruction = "Erstelle aus den die vorliegenden Inhalten eine Relationstabelle wie sie von Richtern / Richterinnen erstellt wird."
    context  = """Die Relationstabelle soll die Parteivorträge zwischen Kläger und Beklagten klar gegenüberstellen. Sie soll die typischen Elemente des Zivilprozesses abbilden. 
    Identifiziere und liste alle relevanten Sachverhaltselemente auf. Fasse den Vortrag jeder Partei zu den jeweiligen Elementen präzise zusammen. Zu jedem Sachverhaltselement, dass extrahiert wurde, soll die entsprechende Passage aus den Dokumenten extrahiert werden. Diese dienen als 'Groundtruth', damit die extrahierten Elemente verifiziert und abgeglichen werden können. Es soll die tatsächliche Textpassage extrahiert werden - keine Zeilen- und Seitenangaben. Fasse den Vortrag jeder Partei wörtlich oder sinngemäß zusammen, ohne eigene Formulierungen hinzuzufügen. Berücksichtige besonders Punkte, die in einem Vortrag vorkommen und im anderen nicht. Markiere diese mit "-", um ein nicht vorhandenes Sachverhaltselement zu markieren. Identifiziere und liste die Beweismittel (z.B. Zeugen, Urkunden, Sachverständigengutachten, Dokumente jeglicher Art), die von jeder Partei für die jeweiligen Sachverhaltselemente angeboten werden."""
    format = "Als Kopf der Relationstabelle wird das dafür zuständige Gericht aus den Dokumenten extrahiert. Die Ausgabe sieht als Beispiel so aus: ## Gericht - **1 C 1/26 – AG Bielefeld** " \
    "Die Ausgabe erfolgt in Tabellenform, mit den Spalten: | Nr. | Sachverhaltselement | Kläger-Vortrag | Kläger-Dokument Passage | Beklagten-Vortrag. | Beklagter-Dokument Passage | Anlagen-Kläger | Anlagen-Beklagter | Verwende die sieben vorgegebenen Spalten in der exakten Reihenfolge."
    audience = "Die Relationstabelle ist zur Strukturierung des Sachverhalts für den Richter vorgesehen, nicht für Laien, den Klägern und Beklagten sowie den beteiligten Rechtsanwälten."
    tone = "Die Tabelle muss streng neutral sein und darf keine rechtliche Wertung, richterliche Einordnung oder Interpretation enthalten. Der Stil ist knapp, objektiv und verwendet typische juristische Formulierungen. Vermeide umgangssprachliche oder ausschmückende Sprache."
    constraints = "Füge keine Informationen hinzu, die nicht explizit in den gegebenen Texten enthalten sind, und ziehe keine Schlussfolgerungen- Der Stil muss sich am Basisdokument und der Relationstabelle nach Streyl (NZM 2021, 805) orientieren. Führe keine rechtlichen Subsumtionen oder Gutachten aus. Verzichte auf die Wiedergabe von Normen und Paragraphen. Die Tabelle soll den reinen Sachverhalt abbilden. Vermeide Formulierungen wie 'Absolut! Hier ist die Relationstabelle, die den Sachverhalt aus Klage und Klageerwiderung strukturiert darstellt:', lediglich die Tabelle soll wiedergegeben werden."
    cot_instruction = """Die Erstellung einer Relationstabelle erfolgt durch folgende Arbeitsschritte:
    1. Lies zunächst beide Dokumente (Klage und Klageerwiderung) vollständig.

    2. Extrahiere die einzelnen, strittigen oder übereinstimmenden Sachverhaltselemente. Konzentriere dich auf die relevanten Sachverhaltselemente. Nicht jeder Nebensatz in der Klage oder Klageerwiderung ist für die Entscheidung von Bedeutung. Nicht vorhandene Sachverhaltselemente werden mit '-' markiert.

    3. Fasse für jedes Element den Vortrag der Klägerseite zusammen. Trage vorhandene Beweismittel in die Spalte 'Beweismittel-Kläger' ein. Markiere nicht vorhandene Beweismittel mit '-'.

    4. Fasse für dasselbe Element den Vortrag der Beklagtenseite zusammen. Trage vorhandene Beweismittel in die Spalte 'Beweismittel-Beklagter' ein. Markiere nicht vorhandene Beweismittel mit '-'.

    5. Trage die gesammelten Informationen in die Tabelle ein.

    """
    one_shot_example = """
    
    Hier erhältst du ein Beispiel wie ein Teil der Relationstabelle hinsichtlich der Sprache, Informationsgehalt und Formatierung auszusehen hat:

    ### I. Räumungsanspruch

# | Kläger                                                                                             | Beklagte                                                                                     |
# |----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
# | Die Parteien schlossen am 01.05.2023 einen Mietvertrag über 700 € (davon 200 € BK-VZ). Zulässig war ausschließlich Wohnnutzung. | — |
# | Herr A. gibt in der Wohnung Yogakurse und überträgt diese via YouTube.                              | Es finden nur gelegentliche Online-Kurse statt, ohne Bezug zur Wohnung. Präsenzkurse nur im Notfall. |
# | Am 12.10.2025 forderte der Kläger die Beklagte auf, die Nutzung zu unterlassen.                     | Es gab weder eine schriftliche noch mündliche Abmahnung. Ein Mahnschreiben wurde nicht vorgelegt. |
# | A beleidigte die Tochter des Klägers, Franziska S., als „alte Vettel“.                             | Es gab einen Streit, aber nicht in dieser Form.                                                |
# | Kündigung des Mietvertrages am 02.12.2025.                                                         | —                                                                                            |
"""
    ground_truth_instruction = """Zu jedem Sachverhaltselement, dass extrahiert wurde, soll die entsprechende Passage aus den Dokumenten extrahiert werden. Diese dienen als 'Groundtruth', damit die extrahierten Elemente verifiziert und abgeglichen werden können. Es soll die tatsächliche Textpassage extrahiert werden - keine Zeilen- und Seitenangaben."""

    prompt = PromptTemplate.from_template(prompt_template)
    output = prompt.invoke({"persona": persona,
                "instruction": instruction,
                "context": context,
                "format": format,
                "audience": audience,
                "tone": tone,
                "constraints": constraints,
                "cot_instruction": cot_instruction,
                "one_shot_example": one_shot_example,
                "num_factual_elements": num_factual_elements,
                "ground_truth_instruction": ground_truth_instruction})
    
    print(output.to_string())

    return output.to_string()

instruction_prompt(25)


