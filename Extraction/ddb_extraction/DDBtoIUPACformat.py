import json
import numpy as np

def data_cleaner(subdicts):
    for i in range(len(subdicts)): #Removes data with pressure values
        try:
            subdict_keys = list(subdicts[i].keys())
            for key in subdict_keys:
                if "P [kPa]" == key:
                    del subdicts[i]
        except:
            continue

    for i in range(len(subdicts)): #Removes data with more than 2 components
        try:
            if len(subdicts[i]["Name"]) > 2:
                del subdicts[i]
        except:
            continue
    
    for i in range(len(subdicts)): #Removes data with no mole fractions
        try:
            subdict_keys = list(subdicts[i].keys())
            if "y1 [mol/mol]" not in subdict_keys:
                del subdicts[i]
            else:
                continue
        except:
            continue

    return subdicts

def NaN_remover(clean_data): #Does what it says on the tin
    for i in range(len(clean_data)):
        mol_fracs = np.array(clean_data[i]["y1 [mol/mol]"])
        clean_data[i]["y1 [mol/mol]"] = mol_fracs[np.logical_not(np.isnan(mol_fracs))]
        clean_data[i]["y1 [mol/mol]"] = list(clean_data[i]["y1 [mol/mol]"])

        if len(clean_data[i]["T [K]"]) > len(clean_data[i]["y1 [mol/mol]"]):
            clean_data[i]["T [K]"] = clean_data[i]["T [K]"][0:len(clean_data[i]["y1 [mol/mol]"])]

    return clean_data

def IUPAC_formatter(clean_data): #Formats the data into a dictionary the same as the IUPAC dataset format
    master_dict = {}
    for i in range(len(clean_data)):
        names = str(clean_data[i]["Name"])
        names = names.replace(", ", " & ")
        names = names.replace("'", "")
        names = names.replace("[", "")
        names = names.replace("]", "")
        mole_fractions = clean_data[i]["y1 [mol/mol]"]
        temp_K = np.array(clean_data[i]["T [K]"])
        temp_C = np.round(temp_K - 273.15, 2)
        temp_C = list(temp_C)
        cas_nums = clean_data[i]["CAS Registry Number"]

        subdict = {"mole fraction:" : mole_fractions,
                     "corresp temps (C):" : temp_C,
                     "cas num:" : cas_nums}

        iupac_dict = {names : subdict}
        master_dict.update(iupac_dict)

    return master_dict

def main():
    with open("Datasets/DDBscrapeddata/full_output.json", "r") as file:
        data = json.load(file)

    subdicts = list(data.values())
    clean_data = data_cleaner(subdicts)
    cleaner_data = NaN_remover(clean_data)
    output = IUPAC_formatter(cleaner_data)

    with open("Datasets/DDBscrapeddata/IUPAC_vap_output.json", "w") as outfile:
        json.dump(output, outfile)

    return output

main()