import pandas as pd
import numpy as np
import re, json
from tqdm import tqdm
from datetime import datetime
import time
import warnings
warnings.filterwarnings("ignore")

def get_compound_pairs(compounds): #Gets the compound pairs and CAS numbers from the metadata
    compound_pair = []

    cas = re.findall(r"\[.*?\]", compounds)
    cas = [x.replace("[", "").replace("]", "") for x in cas]

    for comp in compounds.split("(2)"):
        compound = comp.split(";")[0]
        compound = re.sub(r'\([^)]*\)', '', compound)
        compound = compound.strip()
        compound_pair.append(compound)

    return compound_pair, cas

def table_to_dict(table): #Converts the tables to dictionaries
    dict = {}
    cols = table.columns.tolist()

    for i in range(len(cols)):
        data = table[cols[i]].values.tolist()
        dict.update({cols[i]: data})

    return dict

def iupac_formatter(compounds, temps, cas, mol_fracs):
    output = {}
    names = f"{compounds[0]} & {compounds[1]}"

    subdict = {"mole fraction" : mol_fracs,
            "corresp temps (C)" : temps,
            "cas_nums" : [cas[0], cas[1]]}
    
    iupac_dict = {names : subdict}
    output.update(iupac_dict)
    return output

def compound_pair_dict_builder(comp_tables):
    compound_pair_dict = {}
    for i in range(len(comp_tables)):

        compounds = str(comp_tables[i].iloc[0,0])

        if re.search("(1)", compounds) is None:
            compounds = str(comp_tables[i].iloc[1,0])

        compound_pair, cas = get_compound_pairs(compounds)

        if len(compound_pair) == 2:
            compound_pair_dict.update({str(compound_pair) : cas})

    print(f"No of compound pairs: {len(compound_pair_dict)}")
    return compound_pair_dict

def loc_cleaner(loc):
    newlist = []
    loclist = loc.values.tolist()
    for i in range(len(loclist)):
        newlist.append(loclist[i][0])
    return newlist

def table_searcher(tablelist, substr):
    output = []
    for table in tablelist:
        table = pd.concat([table, table.head()], axis = 0)
        table.reset_index(drop=True, inplace=True)
        cols = table.columns.tolist()

        for i in range(len(cols)):

            if type(substr) == list:
                loc1 = pd.DataFrame(table.iloc[:, i].astype(str).str.contains(f" {substr[0]}", case = False)).drop_duplicates()
                loc2 = pd.DataFrame(table.iloc[:, i].astype(str).str.contains(f" {substr[1]}", case = False)).drop_duplicates()
                loc1 = loc_cleaner(loc1)
                loc2 = loc_cleaner(loc2)
                if True in loc1 and True in loc2:
                    unique = True
                else:
                    unique = False
            else:
                loc = table.iloc[i].str.find(str(substr))
                unique = loc.is_unique

            if unique == True:
                output.append(table)
            else:
                continue
    if len(output) == 0:
        output = None
    return output

def table_pairer(keylist, mf_tables):
    keylist2 = [x.replace("[", "").replace("]", "").replace("'", "") for x in keylist]
    mfdict = {}
    for i in range(len(keylist)):
        keys = keylist2[i].split(", ")
        data_table = table_searcher(mf_tables, keys)
        if data_table == None:
            pass
        else:
            mfdict.update({str(keylist[i]) : data_table})
    print(f"No of mol frac tables paired with names: {len(mfdict)}")

    return mfdict

def generic_list_cleaner(input):
    output = [re.sub(r'\([^)]*\)', '', x) for x in input]
    output = [x.replace(" ", "") for x in output]
    output = [re.sub(r'[^\x00-\x7F]', "", x) for x in output]
    return output

def main(file):
    all_tables = pd.read_html(f"Datasets/IUPAC/SDS/{file}.htm")
    comp_tables = pd.read_html(f"Datasets/IUPAC/SDS/{file}.htm", match = "Components:")
    compound_pair_dict = compound_pair_dict_builder(comp_tables)
    print(f"all tables: {len(all_tables)}")
    print(f"compound pair dict: {len(compound_pair_dict)}")

    keylist = list(compound_pair_dict.keys())
    mf_tables = table_searcher(all_tables, "T∕")
    mf_tables2 = table_searcher(all_tables, "T/")
    if mf_tables == None:
        print("No mole frac tables found")
        mfdict = {}
    else:
        if mf_tables2 != None:
            mf_tables = mf_tables + mf_tables2
        print(f"No of mole frac tables inc duplicates: {len(mf_tables)}")
        mfdict = table_pairer(keylist, mf_tables)

    keylist = list(mfdict.keys())

    output = {}
    for i in range(len(mfdict)):
        key = keylist[i]
        cas = compound_pair_dict[key]
        data = mfdict[key][0].reset_index(drop=True)

        data_dict = {}
        col_data_list = []
        cols = data.columns.tolist()

        for col in cols:
            col_data = data[col].values.tolist()
            col_data_list.append(col_data)
        for j in range(1, len(cols)):
            for k in range(len(col_data_list[0])):
                data_dict.update({col_data_list[j][k] : col_data_list[0][k]}) #mole frac:temp

        temps = list(data_dict.values())
        mole_fracs = list(data_dict.keys())

        unit = "C"
        stop_point = None
        start_point = None
        for j in range(len(temps)):
            if re.search("T∕" or "T/", temps[j]):
                start_point = j+1
            if re.search("T∕K" or "T/K", temps[j]):
                unit = "K"
            if re.search("solubility", temps[j]):
                stop_point = j
                break
        
        temps = temps[start_point:stop_point]
        temps = [re.sub(r'[^0-9.]', "", x) for x in temps]
        temps = generic_list_cleaner(temps)

        if unit == "K":
            temps = [np.round(float(x) - 273.15, 2) for x in temps]
        
        # Mole fraction cleaning
        mole_fracs = mole_fracs[start_point:stop_point]
        mole_fracs = generic_list_cleaner(mole_fracs)

        non_sf_mf = []
        for j in range(len(mole_fracs)):
            if re.search("x10", mole_fracs[j]):
                mf = mole_fracs[j].split("x10")
                mf = float(mf[0]) * (10 ** float(mf[1]))
                non_sf_mf.append(mf)
            else:
                non_sf_mf.append(mole_fracs[j])

        mole_fracs = [re.sub(r'[^0-9.]', "", x) for x in mole_fracs]
        mole_fracs = [float(x) for x in mole_fracs]
        
        key = key.replace("[", "").replace("]", "").replace("'", "")
        key = key.split(", ")
        output.update(iupac_formatter(key, temps, cas, mole_fracs))

    return output

file_names = "SDS102,SDS82_2,SDS86_2,SDS88_1,SDS96_2,SDS82_3,SDS86_3,SDS88_2,SDS96_3,SDS82_4,SDS86_4,SDS88_3,SDS97,SDS105_2,SDS82_5,SDS86_5,SDS88_4,SDS98_1,SDS82_1,SDS86_1,SDS86_6,SDS96_1,SDS98_3".split(",")

for i in range(1, len(file_names)):
    print(datetime.now())
    print(file_names[i])
    try:
        output = main(file_names[i])
        if len(output) == 0:
            print(f"File {file_names[i]} failed")
        else:
            with open(f"Datasets/IUPAC/SDS/{file_names[i]}.json", "w") as outfile:
                json.dump(output, outfile)
            print(f"File {file_names[i]} complete")
        print("-"*70)
    except:
        print(f"File {file_names[i]} failed")
print(datetime.now())