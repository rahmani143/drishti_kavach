import os
import numpy as np


DB_PATH = r"C:\Users\bss10\Desktop\drishti\drishti_kavach\face_recognition\face_database.npz"

def remove_actors_from_database():
    if not os.path.exists(DB_PATH):
        print(f"Error: Database file not found at {DB_PATH}")
        return

    
    with np.load(DB_PATH) as data:
        database = {key: data[key] for key in data.files}
    
    print(f"Current database size: {len(database)} identities.")

    
    actors_to_remove = [
        "abhay_deol", "adil_hussain", "ajay_devgn", "akshaye_khanna", "akshay_kumar", 
        "amitabh_bachchan", "amjad_khan", "amole_gupte", "amol_palekar", "amrish_puri", 
        "anil_kapoor", "annu_kapoor", "anupam_kher", "anushka_shetty", "arshad_warsi", 
        "aruna_irani", "ashish_vidyarthi", "asrani", "atul_kulkarni", "ayushmann_khurrana", 
        "boman_irani", "chiranjeevi", "chunky_panday", "danny_denzongpa", "darsheel_safary", 
        "deepika_padukone", "deepti_naval", "dev_anand", "dharmendra", "dilip_kumar", 
        "dimple_kapadia", "farhan_akhtar", "farida_jalal", "farooq_shaikh", "girish_karnad", 
        "govinda", "gulshan_grover", "hrithik_roshan", "huma_qureshi", "irrfan_khan", 
        "jaspal_bhatti", "jeetendra", "jimmy_sheirgill", "johnny_lever", "kader_khan", 
        "kajol", "kalki_koechlin", "kamal_haasan", "kangana_ranaut", "kay_kay_menon", 
        "konkona_sen_sharma", "kulbhushan_kharbanda", "lara_dutta", "madhavan", "madhuri_dixit", 
        "mammootty", "manoj_bajpayee", "manoj_pahwa", "mehmood", "mita_vashisht", 
        "mithun_chakraborty", "mohanlal", "mohnish_bahl", "mukesh_khanna", "mukul_dev", 
        "nagarjuna_akkineni", "nana_patekar", "nandita_das", "nargis", "naseeruddin_shah", 
        "navin_nischol", "nawazuddin_siddiqui", "neeraj_kabi", "nirupa_roy", "om_puri", 
        "pankaj_kapur", "pankaj_tripathi", "paresh_rawal", "pawan_malhotra", "pooja_bhatt", 
        "prabhas", "prabhu_deva", "prakash_raj", "pran", "prem_chopra", 
        "priyanka_chopra", "raaj_kumar", "radhika_apte", "rahul_bose", "rajat_kapoor", 
        "rajesh_khanna", "rajinikanth", "rajit_kapoor", "rajkummar_rao", "rajpal_yadav", 
        "raj_babbar", "raj_kapoor", "rakhee_gulzar", "ramya_krishnan", "ranbir_kapoor", 
        "randeep_hooda", "rani_mukerji", "ranveer_singh", "ranvir_shorey", "ratna_pathak_shah", 
        "rekha", "richa_chadha", "rishi_kapoor", "riteish_deshmukh", "sachin_khedekar", 
        "saeed_jaffrey", "saif_ali_khan", "salman_khan", "sanjay_dutt", "sanjay_mishra", 
        "shabana_azmi", "shah_rukh_khan", "sharman_joshi", "sharmila_tagore", "shashi_kapoor", 
        "shreyas_talpade", "smita_patil", "soumitra_chatterjee", "sridevi", "sunil_shetty", 
        "sunny_deol", "tabu", "tinnu_anand", "utpal_dutt", "varun_dhawan", 
        "vidya_balan", "vinod_khanna", "waheeda_rehman", "zarina_wahab", "zeenat_aman"
    ]

   
    removed_count = 0
    for name in actors_to_remove:
        if name in database:
            del database[name]
            removed_count += 1

   
    np.savez(DB_PATH, **database)
    print(f"[SUCCESS] Removed {removed_count} actors from the database.")
    print(f"New database size: {len(database)} identities.")


def show_remaining_database_details():
    if not os.path.exists(DB_PATH):
        print(f"Error: Database file not found at {DB_PATH}")
        return

    # Load the database file
    with np.load(DB_PATH) as data:
        remaining_identities = data.files
        
        print("\n--- Remaining Database Details ---")
        print(f"Total people registered: {len(remaining_identities)}\n")
        
        if len(remaining_identities) == 0:
            print("The database is completely empty.")
            return

        # Loop through each person and show their details
        for index, name in enumerate(remaining_identities, start=1):
            vector = data[name]
            print(f"{index}. Name: {name}")
            print(f"   Vector Shape: {vector.shape} (512-dimensional ArcFace embedding)")
            print(f"   Sample Values (First 5 dimensions): {vector[:5]}")
            print("-" * 50)

if __name__ == "__main__":
    show_remaining_database_details()