import base64
import streamlit as st
import pandas as pd
from pulp import *

def optimize_components(budget, max_cpu_pct, max_gpu_pct, max_mb_pct, max_ram_pct, max_psu_pct, max_storage_pct):
    # Încărcarea datelor din fișierele CSV
    cpu_df = pd.read_csv('cpu.csv', delimiter=';')
    gpu_df = pd.read_csv('gpu.csv', delimiter=';')
    mb_df = pd.read_csv('mb.csv', delimiter=';')
    ram_df = pd.read_csv('ram.csv', delimiter=';')
    psu_df = pd.read_csv('psu.csv', delimiter=';')
    storage_df = pd.read_csv('storage.csv', delimiter=';')

    # Crearea modelului de optimizare
    model = LpProblem("Optimizare_Componente_PC", LpMaximize)

    # Variabile
    cpu_vars = LpVariable.dicts("CPU", cpu_df.index, cat='Binary')
    gpu_vars = LpVariable.dicts("GPU", gpu_df.index, cat='Binary')
    mb_vars = LpVariable.dicts("MB", mb_df.index, cat='Binary')
    ram_vars = LpVariable.dicts("RAM", ram_df.index, cat='Binary')
    psu_vars = LpVariable.dicts("PSU", psu_df.index, cat='Binary')
    storage_vars = LpVariable.dicts("Storage", storage_df.index, cat='Binary')

    # Funcția obiectiv - Maximizarea scorului total
    model += lpSum([cpu_df.loc[i, 'score'] * cpu_vars[i] for i in cpu_df.index]) + \
             lpSum([gpu_df.loc[i, 'score'] * gpu_vars[i] for i in gpu_df.index]) + \
             lpSum([ram_df.loc[i, 'score'] * ram_vars[i] for i in ram_df.index]) + \
             lpSum([storage_df.loc[i, 'score'] * storage_vars[i] for i in storage_df.index])

    # Constrângeri
    # Buget
    model += lpSum([cpu_df.loc[i, 'price'] * cpu_vars[i] for i in cpu_df.index]) + \
             lpSum([gpu_df.loc[i, 'price'] * gpu_vars[i] for i in gpu_df.index]) + \
             lpSum([mb_df.loc[i, 'price'] * mb_vars[i] for i in mb_df.index]) + \
             lpSum([ram_df.loc[i, 'price'] * ram_vars[i] for i in ram_df.index]) + \
             lpSum([psu_df.loc[i, 'price'] * psu_vars[i] for i in psu_df.index]) + \
             lpSum([storage_df.loc[i, 'price'] * storage_vars[i] for i in storage_df.index]) <= budget

    # Compatibilitate CPU și Motherboard
    for cpu_index in cpu_df.index:
        for mb_index in mb_df.index:
            if cpu_df.loc[cpu_index, 'socket'] != mb_df.loc[mb_index, 'socket']:
                model += cpu_vars[cpu_index] + mb_vars[mb_index] <= 1

    # Compatibilitate form factor între MB și PSU (pentru carcasa)
    for mb_index in mb_df.index:
        for psu_index in psu_df.index:
            if mb_df.loc[mb_index, 'form_factor'] != psu_df.loc[psu_index, 'type']:
                model += mb_vars[mb_index] + psu_vars[psu_index] <= 1

    # TDP total al CPU și GPU nu trebuie să depășească capacitatea PSU
    model += lpSum([cpu_df.loc[i, 'tdp'] * cpu_vars[i] for i in cpu_df.index]) + \
             lpSum([gpu_df.loc[i, 'tdp'] * gpu_vars[i] for i in gpu_df.index]) <= \
             lpSum([psu_df.loc[i, 'wattage'] * psu_vars[i] for i in psu_df.index]), "Total_TDP_Less_Than_PSU_Capacity"

    # Puterea recomandată pentru GPU trebuie să fie acoperită de PSU
    model += lpSum([gpu_df.loc[i, 'recomendedw'] * gpu_vars[i] for i in gpu_df.index]) <= \
             lpSum([psu_df.loc[i, 'wattage'] * psu_vars[i] for i in psu_df.index]), "GPU_Recomennded_PSU_Capacity"

    # Capacitatea totală de RAM să nu depășească capacitatea maximă a plăcii de bază selectate
    model += lpSum([ram_df.loc[ram_index, 'gb'] * ram_df.loc[ram_index, 'modules'] * ram_vars[ram_index] for ram_index in ram_df.index]) <= \
             lpSum([mb_df.loc[mb_index, 'max_memory'] * mb_vars[mb_index] for mb_index in mb_df.index]), "Max_RAM_Capacity"

    # Numărul de module RAM nu trebuie sa depaseasca numarul de sloturi de pe placa de bază
    model += lpSum([ ram_df.loc[ram_index, 'modules'] * ram_vars[ram_index] for ram_index in ram_df.index]) <= \
             lpSum([mb_df.loc[mb_index, 'memory_slots'] * mb_vars[mb_index] for mb_index in mb_df.index]), "Max_RAM_Modules"

    # Numar componente
    model += lpSum([cpu_vars[i] for i in cpu_df.index]) == 1, "Exact_one_CPU"
    model += lpSum([gpu_vars[i] for i in gpu_df.index]) == 1, "Exact_one_GPU"
    model += lpSum([mb_vars[i] for i in mb_df.index]) == 1, "Exact_one_MB"
    model += lpSum([psu_vars[i] for i in psu_df.index]) == 1, "Exact_one_PSU"
    model += lpSum([storage_vars[i] for i in storage_df.index]) == 1, "Exact_one_Storage"
    model += lpSum([ram_vars[i] for i in ram_df.index]) == 1, "Exact_one_RAM"

    # Optional: Constrangeri pentru pretul maxim din buget pentru fiecare componenta
    model += lpSum([cpu_df.loc[i, 'price'] * cpu_vars[i] for i in cpu_df.index]) <= budget * (max_cpu_pct / 100), "MaxCPUPriceConstraint"
    model += lpSum([gpu_df.loc[i, 'price'] * gpu_vars[i] for i in gpu_df.index]) <= budget * (max_gpu_pct / 100), "MaxGPUPriceConstraint"
    model += lpSum([mb_df.loc[i, 'price'] * mb_vars[i] for i in mb_df.index]) <= budget * (max_mb_pct / 100), "MaxMBPriceConstraint"
    model += lpSum([ram_df.loc[i, 'price'] * ram_vars[i] for i in ram_df.index]) <= budget * (max_ram_pct / 100), "MaxRAMPriceConstraint"
    model += lpSum([psu_df.loc[i, 'price'] * psu_vars[i] for i in psu_df.index]) <= budget * (max_psu_pct / 100), "MaxPSUPriceConstraint"
    model += lpSum([storage_df.loc[i, 'price'] * storage_vars[i] for i in storage_df.index]) <= budget * (max_storage_pct / 100), "MaxStoragePriceConstraint"

    model.solve()

    results = {}
    total_score = 0
    total_price = 0

    # Rezultate
    if LpStatus[model.status] == 'Optimal':
        for v in model.variables():
            if v.varValue > 0:
                category, index = v.name.split("_", 1)
                index = int(index)
                if category == "CPU":
                    results["CPU"] = cpu_df.loc[[index]]
                    total_score += cpu_df.loc[index, 'score']
                    total_price += cpu_df.loc[index, 'price']
                elif category == "GPU":
                    results["GPU"] = gpu_df.loc[[index]]
                    total_score += gpu_df.loc[index, 'score']
                    total_price += gpu_df.loc[index, 'price']
                elif category == "MB":
                    results["Motherboard"] = mb_df.loc[[index]]
                    total_price += mb_df.loc[index, 'price']
                elif category == "RAM":
                    results["RAM"] = ram_df.loc[[index]]
                    total_score += ram_df.loc[index, 'score']
                    total_price += ram_df.loc[index, 'price']
                elif category == "PSU":
                    results["PSU"] = psu_df.loc[[index]]
                    total_price += psu_df.loc[index, 'price']
                elif category == "Storage":
                    results["Storage"] = storage_df.loc[[index]]
                    total_score += storage_df.loc[index, 'score']
                    total_price += storage_df.loc[index, 'price']
    else:
        st.error("Modelul nu a fost rezolvat cu succes. Status: " + LpStatus[model.status])
    return results, total_score, total_price

# Meniu lateral pentru navigare
with st.sidebar:
    selected_page = st.radio("Navigare", ["Optimizare Componente", "Descriere Problema"])

if selected_page == "Optimizare Componente":
    st.title("Optimizare Componente Calculator")

    # Input pentru buget
    budget = st.number_input("Introduceți bugetul:", min_value=0,value=2000)

    # Toggle pentru a decide dacă să afișăm inputurile pentru procente
    modify_percentages = st.checkbox("Modificați procentele pentru bugetul componentelor")

    # Dacă utilizatorul alege să modifice procentele
    if modify_percentages:
        #Modificare procent maxim din pret pentru componente
        max_cpu_pct = st.number_input("Procent maxim din buget pentru CPU:", 0, 100, 100)
        max_gpu_pct = st.number_input("Procent maxim din buget pentru GPU:", 0, 100, 100)
        max_mb_pct = st.number_input("Procent maxim din buget pentru Motherboard:", 0, 100, 100)
        max_ram_pct = st.number_input("Procent maxim din buget pentru RAM:", 0, 100, 100)
        max_psu_pct = st.number_input("Procent maxim din buget pentru PSU:", 0, 100, 100)
        max_storage_pct = st.number_input("Procent maxim din buget pentru Storage:", 0, 100, 100)
    else:
        # Setăm procentele la 100% dacă utilizatorul nu dorește să le modifice
        max_cpu_pct, max_gpu_pct, max_mb_pct, max_ram_pct, max_psu_pct, max_storage_pct = 100, 100, 100, 100, 100, 100


    if st.button("Optimizează"):
        result_dict, total_score, total_price = optimize_components(budget, max_cpu_pct, max_gpu_pct, max_mb_pct, max_ram_pct, max_psu_pct, max_storage_pct)
        for component_type, df in result_dict.items():
            st.subheader(component_type)
            st.dataframe(df)

        # Afișează scorul și prețul total
        st.write(f"Scorul total: {total_score}")
        st.write(f" Prețul total: {total_price}$")

elif selected_page == "Descriere Problema":

    st.title("Descrierea problemei si resurse")
    st.markdown("---")
    pdf_file_path = 'oro.pdf'

    # Verificați dacă fișierul există
    if os.path.exists(pdf_file_path):
        # Deschiderea și afișarea PDF-ului
        with open(pdf_file_path, "rb") as pdf_file:
            base64_pdf = base64.b64encode(pdf_file.read()).decode('utf-8')
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'

            # Afișarea PDF-ului
            st.markdown(pdf_display, unsafe_allow_html=True)

            # Afișarea unui buton de descărcare pentru PDF
            pdf_file.seek(0)
            st.download_button(
                label="Descărcați PDF-ul",
                data=pdf_file,
                file_name=pdf_file_path,
                mime='application/octet-stream',
            )
    else:
        st.error("Fișierul PDF nu a fost găsit.")

    st.markdown("---")
    data_source_url ="https://github.com/docyx/pc-part-dataset/tree/main"
    st.write("Sursa datelor utilizate")
    st.link_button("Deschide",data_source_url)



