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
    "In den vorliegenden Dokumenten liegen mindestens {num_factual_elements} Sachverhaltselemente vor. Diese Zahl an Elementen muss mindestens in der Relationstabelle vorkommen." 


    persona = "Du bist ein Richter an einem deutschen Gericht. Du wirst mit Streitigkeiten die sich innerhalb des deutschen Zivilrechts bewegen betraut."
    instruction = "Erstelle aus den die vorliegenden Inhalten eine Relationstabelle wie sie von Richtern / Richterinnen erstellt wird."
    context  = """Die Relationstabelle soll die Parteivorträge zwischen Kläger und Beklagten klar gegenüberstellen. Sie soll die typischen Elemente des Zivilprozesses abbilden. 
    Identifiziere und liste alle relevanten Sachverhaltselemente auf. Fasse den Vortrag jeder Partei zu den jeweiligen Elementen präzise und knapp zusammen. Fasse den Vortrag jeder Partei wörtlich oder sinngemäß so knapp wie möglich zusammen, ohne eigene Formulierungen hinzuzufügen. Berücksichtige besonders Punkte, die in einem Vortrag vorkommen und im anderen nicht. Markiere diese mit "-", um ein nicht vorhandenes Sachverhaltselement zu markieren. Identifiziere und liste die Beweismittel (z.B. Zeugen, Urkunden, Sachverständigengutachten, Dokumente jeglicher Art), die von jeder Partei für die jeweiligen Sachverhaltselemente angeboten werden."""
    format = "Als Kopf der Relationstabelle wird das dafür zuständige Gericht aus den Dokumenten extrahiert. Die Ausgabe sieht als Beispiel so aus: ## Gericht - **1 C 1/26 – AG Bielefeld** " \
    "Die Ausgabe erfolgt in Tabellenform, mit den Spalten: | Nr. | Sachverhaltselement | Kläger-Vortrag | Beklagten-Vortrag. | Anlagen-Kläger | Anlagen-Beklagter | Verwende die fünf vorgegebenen Spalten in der exakten Reihenfolge."
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
    
#     few_shot = """
#     # Basisdokument 

# ## Gericht
# **1 C 1/26 – AG Bielefeld**

# ---

# ## Parteien

# | Rolle          | Person                         | Vertreter                         |
# |----------------|--------------------------------|-----------------------------------|
# | **Klagepartei**| Hubert S., Linz                | RA Brenner, Koblenz               |
# | **Beklagte**   | Julia G., Bielefeld             | RAin Odenwald, Stuttgart          |

# ---

# ## Anträge

# | Datum       | Antrag Klagepartei                                                                 | Antrag Beklagtenpartei                                        |
# |-------------|------------------------------------------------------------------------------------|---------------------------------------------------------------|
# | 02.01.2026  | 1. Räumung und Herausgabe der Wohnung Horststr. 47, Bielefeld.<br>2. Zahlung 353,68 € + Zinsen ab 15.10.2025. | —                                                             |
# | 20.01.2026  | —                                                                                  | Klageabweisung, hilfsweise Räumungsfrist                      |

# ---

# ## Streitwert
# **6.353,68 Euro**

# ---

# ## Vorgerichtliche Einigungsversuche
# Am **19.12.2025** fand eine Besprechung mit der Beklagten und ihrer Prozessbevollmächtigten statt.  
# Eine gütliche Einigung konnte nicht erzielt werden.

# ---

# ## Gegenstand des Rechtsstreits
# Der Kläger nimmt die Beklagte auf **Räumung** und **Betriebskostennachzahlung** in Anspruch.

# ---

# ## Sachverhalt

# ### Einleitung

# | Kläger                                                                                             | Beklagte                                                                                     |
# |----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
# | Der Kläger ist hochbetagt. Er überließ der Beklagten 2023 die Wohnung, obwohl er eigentlich nicht mehr vermieten wollte. Zunächst verlief das Mietverhältnis reibungslos. Mit dem Einzug von Markus A. (Lebensgefährte der Beklagten) kam es zu Unfrieden und angeblicher gewerblicher Nutzung. | Richtig ist, dass es Spannungen gibt. Diese gehen aber nicht von der Beklagten und ihrem Lebensgefährten aus, sondern vom Kläger. Herr A. und sein Sohn Bruno verweigerten unentgeltliche Hilfsdienste. |

# ---

# ### I. Räumungsanspruch

# | Kläger                                                                                             | Beklagte                                                                                     |
# |----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
# | Die Parteien schlossen am 01.05.2023 einen Mietvertrag über 700 € (davon 200 € BK-VZ). Zulässig war ausschließlich Wohnnutzung. | — |
# | Herr A. gibt in der Wohnung Yogakurse und überträgt diese via YouTube.                              | Es finden nur gelegentliche Online-Kurse statt, ohne Bezug zur Wohnung. Präsenzkurse nur im Notfall. |
# | Am 12.10.2025 forderte der Kläger die Beklagte auf, die Nutzung zu unterlassen.                     | Es gab weder eine schriftliche noch mündliche Abmahnung. Ein Mahnschreiben wurde nicht vorgelegt. |
# | A beleidigte die Tochter des Klägers, Franziska S., als „alte Vettel“.                             | Es gab einen Streit, aber nicht in dieser Form.                                                |
# | Kündigung des Mietvertrages am 02.12.2025.                                                         | —                                                                                            |

# ---

# ### II. Betriebskostennachzahlung

# | Kläger                                                                                             | Beklagte                                                                                     |
# |----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
# | Beklagte schuldet 353,68 € aus der Abrechnung 2024. Zustellung am 20.8.2025, Zahlungsfrist bis 15.10.2025. | Abrechnung erstmals mit Klage erhalten. Heizkostenzähler defekt, Hausmeister nicht tätig, Stromabrechnung fehlerhaft. |
# | Abrechnung korrekt, Heizkostenzähler funktionierten.                                               | Heizkostenzähler defekt → Einwendungen.                                                      |
# | Hausmeisterstunden korrekt abgerechnet, Strom Garage separat gezählt.                             | Hausmeister nie tätig. Stromzähler versorgt auch Privatgarage → Kosten nicht umlagefähig.     |

# ---

# ### III. Räumungsschutz

# | Kläger                                                                                             | Beklagte                                                                                     |
# |----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
# | Bestreitet mit Nichtwissen.                                                                        | Für den Fall der Räumung: Frist, da Sohn Bruno A. im Juli 2026 Examen schreibt.              |

# ---

# ## Rechtsausführungen

# | Kläger                                                                                             | Beklagte                                                                                     |
# |----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
# | Online-Kurse rechtfertigen Kündigung. Ebenso die zweimaligen Präsenzkurse.                         | Online-Kurse ohne Außenwirkung, keine Pflichtverletzung. Wenige Präsenzkurse nicht erheblich. |
# | Betriebskostenabrechnung korrekt, Bestreiten der Beklagten unglaubwürdig.                          | —                                                                                            |

# ---

# ## Ausführungen zur Beweisaufnahme

# | Kläger                                                                                             | Beklagte |
# |----------------------------------------------------------------------------------------------------|----------|
# | Zeuge A wurde wegen Ruhestörung angezeigt und drohte dem Kläger Rache an.                         | —        |

# ---

# ## Hinweise und Erläuterungen

# - *Kursiv*: Neuer Vortrag nach dem ersten Schriftsatz.  
# - *Durchgestrichen*: geänderter Vortrag.  
# - Gliederung dient der **historischen Nachvollziehbarkeit** des Vortrags.  
# """

    prompt = PromptTemplate.from_template(prompt_template)
    output = prompt.invoke({"persona": persona,
                "instruction": instruction,
                "context": context,
                "format": format,
                "audience": audience,
                "tone": tone,
                "constraints": constraints,
                "cot_instruction": cot_instruction,
                #"few_shot": few_shot,
                "num_factual_elements": num_factual_elements})

    return output.to_string()

def print_prompt() -> None:
    with open("prompt.txt", "w", encoding="utf-8") as file:
        prompt = instruction_prompt(num_factual_elements=25)
        file.write(prompt)
