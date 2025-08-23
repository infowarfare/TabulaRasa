import os
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai.types import CreateCachedContentConfig, Content, Part
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from prompt import instruction_prompt, factual_elements_prompt
from pathlib import Path
from collections import Counter
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# folder for uploaded files
file_path = "court_files"

# Environment variables
load_dotenv(dotenv_path=".env")
api_key=os.getenv("GOOGLE_API_KEY")

def llm_response_to_doc(response: str) -> str:
    pdf_filename = "llm_response.pdf"
    story = []

    styles = getSampleStyleSheet()
    paragraph = Paragraph(response, styles['Normal'])
    story.append(paragraph)

    doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
    doc.build(story)


def upload_files_to_cache(client) -> str:

    uploaded_files = []
    parts = []

    file_paths = [os.path.join(file_path, f) for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f))]
    
    for path in file_paths:
        path = path.replace("\\", "\\\\")
        file = client.files.upload(file=path)
        while file.state.name == 'PROCESSING':
            time.sleep(2)
            file = client.files.get(name=file.name)
        uploaded_files.append(file)

    for client_file in uploaded_files:
        parts.append(Part.from_uri(file_uri=client_file.uri, mime_type=client_file.mime_type))

    contents = [Content(role="user", parts=parts,)]

    model_name = "gemini-2.5-flash-lite"
    cache = client.caches.create(
        model=model_name,
        config=CreateCachedContentConfig(
            display_name='cached_documents',
            system_instruction=(
                """Du bist ein Richter / eine Richterin an einem deutschen Gericht, welches der ordentlichen Gerichtsbarkeit unterliegt, gemäß des deutschen Gerichtsverfassungsgesetzes (GVG).
                Dein Rechtsgebiet ist das deutsche Zivilrecht welches dem Zivilprozessrecht unterliegt. Das Zivilprozessrecht sieht vor, Ansprüche und Rechte zu ermitteln und durchzusetzen. Im Zivilprozess werden rechtliche Streitigkeiten
                zwischen Kläger und Beklagten verbindlich klären. Vor Gericht erfolgt ein sogenanntes Erkenntnisverfahren, in dem der Sachverhalt aufgeklärt und rechtlich beurteilt wird. Zu Beginn des Verfahrens erfolgen die Parteienvorträge durch Kläger und Beklagter.
                Deine Aufgabe als Richter/in besteht nun darin, den Sachverhalt übersichtlich zu strukturieren, in dem eine Relationstabelle erstellt werden soll.
                """
            ),
            contents=contents,
            ttl="300s",
        )
    )

    return cache.name
    

def generate_answer(cache_name: str, number_of_responses: int) -> str:

    # prompt_template = """Erstelle aus den gegebenen Inhalten von Klage und Klageerwiderung eine Relationstabelle, wie sie im deutschen Zivilprozess von einem Richter erstellt wird.  

    # Die Tabelle soll:  
    # den Parteivortrag der Kläger- und Beklagtenseite klar gegenüberstellen,  
    # ohne Wertung oder richterliche Einordnung erfolgen (also so, wie sie vor richterlicher Bearbeitung im Basisdokument stünde),  
    # nach dem Stil des Basisdokuments und der Relationstabelle gemäß dem Aufsatz von Streyl (NZM 2021, 805) gestaltet sein,  
    # die typischen Elemente des Zivilprozesses abbilden,  
    # in Tabellenform dargestellt sein, mit den Spalten: Nr. | Sachverhaltselement | Kläger-Vortrag | Beklagten-Vortrag.  

    # Ziel ist eine neutrale, strukturierte Übersicht, die als Grundlage für den Tatbestand des Urteils oder das elektronische Basisdokument dienen könnte."""

    #prompt = PromptTemplate(template=prompt_template)

    #prompt.save("base_prompt.json")

    model_name = "gemini-2.5-flash-lite"
    # Query with LangChain
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        cached_content=cache_name,
    )

    # factual_prompt = factual_elements_prompt()
    
    # elements_found = []
    # factual_elements_count = 15 # default number
    # # Self-consistency
    # for i in range(number_of_responses):
    #     count_message = HumanMessage(content=factual_prompt)
    #     response = llm.invoke([count_message])
    #     number_of_elements = int(response.content)
    #     elements_found.append(number_of_elements)

    # if elements_found:
    #     answer_count = Counter(elements_found)
    #     most_consistent_answer, anzahl = answer_count.most_common(1)[0]
    #     factual_elements_count = most_consistent_answer

    # Die Prompt erhält eine Vorgabe, wie wieviele Sachverhaltselemente in den Dokumenten gefunden werden sollen. Es handelt sich hierbei um einen heuristischen Wert.     
    prompt = instruction_prompt(num_factual_elements=25)
    message = HumanMessage(content=prompt)
    response = llm.invoke([message])
    
    st.write(response.content)
    #llm_response_to_doc(response.content)


def main():
    st.set_page_config("Chat PDF")
    st.header("Legal Assistant with Gemini ⚖️")

    exec_button_clicked = st.button("Start Assistant", icon="▶")
    if exec_button_clicked:
         with st.spinner("Start assistant...", show_time=True):
            client = genai.Client(api_key=api_key)
            if client:
                st.success("Agent ready!", icon="🙋🏻")
         with st.spinner("Sending files to Assistant..."):
            cache_name = upload_files_to_cache(client)
            if cache_name:
                st.success("Files received!", icon="🗂️")
         with st.spinner("Generate table..."):
            generate_answer(cache_name, 3)
         st.success("Done!", icon="✅")
        
       

    with st.sidebar:
        st.title("Menu:")
        files = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True, type=['pdf']) # pdf only
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                for f in files:
                    save_path = Path(file_path, f.name)
                    with open(save_path, mode='wb') as file:
                        file.write(f.getvalue())

                    if save_path.exists():
                        st.success("Files successfully saved!")
                st.success("Done")



if __name__ == "__main__":
    main()