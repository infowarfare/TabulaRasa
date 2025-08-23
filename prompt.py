from langchain.prompts import PromptTemplate

def factual_elements_prompt() -> str:

    prompt_template = """Zähle die Sachverhaltselemente die in Dokument vorkommen. Gebe diese Zahl als einzige Ganzzahl aus"""
    return prompt_template

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
    "In der vorliegenden Dokumenten liegen insgesamt {num_factual_elements} Sachverhaltselemente vor. Diese Zahl an Elementen muss in der Relationstabelle vorkommen." 


    persona = "Du bist ein Richter an einem deutschen Gericht. Du wirst mit Streitigkeiten die sich innerhalb des deutschen Zivilrechts bewegen betraut."
    instruction = "Erstelle aus den die vorliegenden Inhalten eine Relationstabelle wie sie von Richtern / Richterinnen erstellt wird."
    context  = """Die Relationstabelle soll die Parteivorträge zwischen Kläger und Beklagten klar gegenüberstellen. Sie soll die typischen Elemente des Zivilprozesses abbilden. 
    Identifiziere und liste alle relevanten Sachverhaltselemente auf. Fasse den Vortrag jeder Partei zu den jeweiligen Elementen präzise und knapp zusammen. Fasse den Vortrag jeder Partei wörtlich oder sinngemäß so knapp wie möglich zusammen, ohne eigene Formulierungen hinzuzufügen. Berücksichtige besonders Punkte, die in einem Vortrag vorkommen und im anderen nicht. Markiere diese mit "-", um ein nicht vorhandenes Sachverhaltselement zu markieren. Identifiziere und liste die Beweismittel (z.B. Zeugen, Urkunden, Sachverständigengutachten, Dokumente jeglicher Art), die von jeder Partei für die jeweiligen Sachverhaltselemente angeboten werden."""
    format = "Die Ausgabe erfolgt in Tabellenform, mit den Spalten: | Nr. | Sachverhaltselement | Kläger-Vortrag | Beklagten-Vortrag. | Beweismittel-Kläger | Beweismittel-Beklagter | Verwende die fünf vorgegebenen Spalten in der exakten Reihenfolge."
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

    prompt = PromptTemplate.from_template(prompt_template)
    output = prompt.invoke({"persona": persona,
                "instruction": instruction,
                "context": context,
                "format": format,
                "audience": audience,
                "tone": tone,
                "constraints": constraints,
                "cot_instruction": cot_instruction,
                "num_factual_elements": num_factual_elements})

    return output.to_string()