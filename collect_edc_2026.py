#!/usr/bin/env python3
"""Collect Spotify audio features for all EDC Las Vegas 2026 artists."""

import os
import sys
import time
import requests
from util import get_spotify_token, get_artist_audio_features

def load_env_file(filepath=".env"):
    if os.path.exists(filepath):
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if line and "=" in line and not line.startswith("#"):
                    key, val = line.split("=", 1)
                    os.environ[key] = val

token = None

def initialize_token():
    global token
    load_env_file()
    client_id = os.getenv("CLIENT_ID")
    client_secret = os.getenv("CLIENT_SECRET")
    if not client_id or not client_secret:
        raise ValueError("CLIENT_ID and CLIENT_SECRET must be set in .env file")
    print("Getting new Spotify access token...")
    token_data = get_spotify_token(client_id, client_secret)
    token = token_data['access_token']
    print("Token obtained successfully!")

def collect_artist(artist_name, output_file):
    global token
    if token is None:
        initialize_token()
    try:
        get_artist_audio_features(artist_name, output_file, token)
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            print("Token expired, refreshing...")
            initialize_token()
            get_artist_audio_features(artist_name, output_file, token)
        elif e.response.status_code == 429:
            retry_after = int(e.response.headers.get("Retry-After", 30))
            retry_after = min(retry_after, 120)  # Cap wait at 2 minutes
            print(f"Rate limited. Waiting {retry_after} seconds...")
            time.sleep(retry_after)
            get_artist_audio_features(artist_name, output_file, token)
        else:
            raise

# (spotify_search_name, csv_filename)
# Already-collected artists are excluded; their files already exist in data/
EDC_2026_ARTISTS = [
    ("1991", "1991"),
    ("999999999", "999999999"),
    ("A.M.C", "amc"),
    ("MC Phantom", "mc_phantom"),
    ("Abana", "abana"),
    ("Juliet Mendoza", "juliet_mendoza"),
    ("Above & Beyond", "above_and_beyond"),
    ("Adiel", "adiel"),
    ("Adrián Mills", "adrian_mills"),
    ("Adriatique", "adriatique"),
    ("Æon:Mode", "aeon_mode"),
    ("Ahee", "ahee"),
    ("Liquid Stranger", "liquid_stranger"),
    ("Ahmed Spins", "ahmed_spins"),
    ("Alves", "alves"),
    ("Alyssa Jolee", "alyssa_jolee"),
    ("Anastazja", "anastazja"),
    ("Andrew Rayel", "andrew_rayel"),
    ("ANNA", "anna"),
    ("AR/CO", "ar_co"),
    ("Argy", "argy"),
    ("Armin van Buuren", "armin_van_buuren"),
    ("Astrix", "astrix"),
    ("ATLiens", "atliens"),
    ("Audiofreq", "audiofreq"),
    ("Code Black", "code_black"),
    ("Toneshifterz", "toneshifterz"),
    ("Avalon Emerson", "avalon_emerson"),
    ("Dennett", "dennett"),
    ("Bad Boombox", "bad_boombox"),
    ("Ollie Lishman", "ollie_lishman"),
    ("Bashkka", "bashkka"),
    ("Sedef Adasï", "sedef_adasi"),
    ("Baugruppe90", "baugruppe90"),
    ("Beltran", "beltran"),
    ("Simas", "simas"),
    ("Benwal", "benwal"),
    ("Billy Gillies", "billy_gillies"),
    ("Black Tiger Sex Machine", "black_tiger_sex_machine"),
    ("Bolo", "bolo"),
    ("Boogie T", "boogie_t"),
    ("Distinct Motive", "distinct_motive"),
    ("Bou", "bou"),
    ("Boys Noize", "boys_noize"),
    ("Bullet Tooth", "bullet_tooth"),
    ("The Carry Nation", "the_carry_nation"),
    ("Cassian", "cassian"),
    ("The Chainsmokers", "the_chainsmokers"),
    ("Charlotte de Witte", "charlotte_de_witte"),
    ("Chris Lorenzo", "chris_lorenzo"),
    ("Chris Stussy", "chris_stussy"),
    ("CID", "cid"),
    ("Clawz", "clawz"),
    ("Cloonee", "cloonee"),
    ("Cloudy", "cloudy"),
    ("Club Angel", "club_angel"),
    ("Cold Blue", "cold_blue"),
    ("Confidence Man", "confidence_man"),
    ("Cosmic Gate", "cosmic_gate"),
    ("Cristoph", "cristoph"),
    ("Culture Shock", "culture_shock"),
    ("CUTDWN", "cutdwn"),
    ("Cyclops", "cyclops"),
    ("Da Tweekaz", "da_tweekaz"),
    ("Dabin", "dabin"),
    ("Darren Porter", "darren_porter"),
    ("Darude", "darude"),
    ("Dead X", "dead_x"),
    ("Deathpact", "deathpact"),
    ("Delta Heavy", "delta_heavy"),
    ("Discip", "discip"),
    ("DJ Gigola", "dj_gigola"),
    ("DJ Isaac", "dj_isaac"),
    ("DJ Mandy", "dj_mandy"),
    ("DJ Tennis", "dj_tennis"),
    ("Chloé Caillet", "chloe_caillet"),
    ("Red Axes", "red_axes"),
    ("Doctor P", "doctor_p"),
    ("Flux Pavilion", "flux_pavilion"),
    ("Funtcase", "funtcase"),
    ("Dømina", "domina"),
    ("Dreya V", "dreya_v"),
    ("Dyen", "dyen"),
    ("EazyBaked", "eazybaked"),
    ("Eli & Fur", "eli_and_fur"),
    ("Eli Brown", "eli_brown"),
    ("Eptic", "eptic"),
    ("Space Laces", "space_laces"),
    ("Fallen", "fallen"),
    ("MC Dino", "mc_dino"),
    ("FISHER", "fisher"),
    ("Frankie Bones", "frankie_bones"),
    ("Funk Tribu", "funk_tribu"),
    ("Gareth Emery", "gareth_emery"),
    ("Getter", "getter"),
    ("Ghengar", "ghengar"),
    ("Gorillat", "gorillat"),
    ("Gravagerz", "gravagerz"),
    ("GRAVEDGR", "gravedgr"),
    ("HAAi", "haai"),
    ("Luke Alessi", "luke_alessi"),
    ("Hamdi", "hamdi"),
    ("Hannah Laing", "hannah_laing"),
    ("Hardwell", "hardwell"),
    ("Hayla", "hayla"),
    ("Heidi Lawden", "heidi_lawden"),
    ("Masha Mar", "masha_mar"),
    ("Heyz", "heyz"),
    ("HNTR", "hntr"),
    ("HOL!", "hol"),
    ("Holy Priest", "holy_priest"),
    ("Hybrid Minds", "hybrid_minds"),
    ("I Hate Models", "i_hate_models"),
    ("Ilan Bluestone", "ilan_bluestone"),
    ("Indira Paganotto", "indira_paganotto"),
    ("Infekt", "infekt"),
    ("Samplifire", "samplifire"),
    ("Innellea", "innellea"),
    ("Interplanetary Criminal", "interplanetary_criminal"),
    ("Isabella", "isabella"),
    ("Jackie Hollander", "jackie_hollander"),
    ("John Summit", "john_summit"),
    ("Joseph Capriati", "joseph_capriati"),
    ("Josh Baker", "josh_baker"),
    ("Kai Wachi", "kai_wachi"),
    ("Kaskade", "kaskade"),
    ("KETTAMA", "kettama"),
    ("Kevin de Vries", "kevin_de_vries"),
    ("KI/KI", "ki_ki"),
    ("Kinahau", "kinahau"),
    ("Klangkuenstler", "klangkuenstler"),
    ("Klo", "klo"),
    ("Korolova", "korolova"),
    ("KREAM", "kream"),
    ("Kuko", "kuko"),
    ("Johannes Schuster", "johannes_schuster"),
    ("Lady Faith", "lady_faith"),
    ("LNY TNZ", "lny_tnz"),
    ("Laidback Luke", "laidback_luke"),
    ("Chuckie", "chuckie"),
    ("Layton Giordani", "layton_giordani"),
    ("Level Up", "level_up"),
    ("Lilly Palmer", "lilly_palmer"),
    ("Linska", "linska"),
    ("LU.RE", "lu_re"),
    ("Luciano", "luciano"),
    ("Luke Dean", "luke_dean"),
    ("Luuk van Dijk", "luuk_van_dijk"),
    ("Maddix", "maddix"),
    ("MALUGI", "malugi"),
    ("Maria Healy", "maria_healy"),
    ("Martin Garrix", "martin_garrix"),
    ("Mary Droppinz", "mary_droppinz"),
    ("Massano", "massano"),
    ("Massimiliano Pagliara", "massimiliano_pagliara"),
    ("Mathame", "mathame"),
    ("Matty Ralph", "matty_ralph"),
    ("Max Dean", "max_dean"),
    ("MCR-T", "mcr_t"),
    ("Meduza", "meduza"),
    ("Mëstiza", "mestiza"),
    ("Mink", "mink"),
    ("Mish", "mish"),
    ("Morgan Seatree", "morgan_seatree"),
    ("MPH", "mph"),
    ("Murphy's Law", "murphys_law"),
    ("Muzz", "muzz"),
    ("Nico Moreno", "nico_moreno"),
    ("Nightstalker", "nightstalker"),
    ("Noizu", "noizu"),
    ("Nostalgix", "nostalgix"),
    ("Notion", "notion"),
    ("Obskür", "obskur"),
    ("Omar+", "omar_plus"),
    ("OMNOM", "omnom"),
    ("The Outlaw", "the_outlaw"),
    ("Paramida", "paramida"),
    ("Paul Oakenfold", "paul_oakenfold"),
    ("Paul van Dyk", "paul_van_dyk"),
    ("PEEKABOO", "peekaboo"),
    ("Pegassi", "pegassi"),
    ("Peggy Gou", "peggy_gou"),
    ("Player Dave", "player_dave"),
    ("Porter Robinson", "porter_robinson"),
    ("The Prodigy", "the_prodigy"),
    ("Prospa", "prospa"),
    ("The Purge", "the_purge"),
    ("Ray Volpe", "ray_volpe"),
    ("Rebekah", "rebekah"),
    ("Rebūke", "rebuke"),
    ("Restricted", "restricted"),
    ("RIOT", "riot"),
    ("Rob Gee", "rob_gee"),
    ("Lenny Dee", "lenny_dee"),
    ("Roddy Lima", "roddy_lima"),
    ("Rooler", "rooler"),
    ("Røz", "roz"),
    ("The Saints", "the_saints"),
    ("Salute", "salute"),
    ("Sama' Abdulhadi", "sama_abdulhadi"),
    ("Sammy Virji", "sammy_virji"),
    ("San Holo", "san_holo"),
    ("San Pacho", "san_pacho"),
    ("Sarah de Warren", "sarah_de_warren"),
    ("Serafina", "serafina"),
    ("Seven Lions", "seven_lions"),
    ("Shingo Nakamura", "shingo_nakamura"),
    ("Ship Wrek", "ship_wrek"),
    ("Sidney Charles", "sidney_charles"),
    ("Bushbaby", "bushbaby"),
    ("Sihk", "sihk"),
    ("Silva Bumpa", "silva_bumpa"),
    ("Silvie Loto", "silvie_loto"),
    ("Sippy", "sippy"),
    ("Skream", "skream"),
    ("Slamm", "slamm"),
    ("Slugg", "slugg"),
    ("Sofi Tukker", "sofi_tukker"),
    ("Solomun", "solomun"),
    ("Spray", "spray"),
    ("Stacy Christine", "stacy_christine"),
    ("Stan Christ", "stan_christ"),
    ("Steve Aoki", "steve_aoki"),
    ("Sub Focus", "sub_focus"),
    ("Sub Zero Project", "sub_zero_project"),
    ("Subtronics", "subtronics"),
    ("Superstrings", "superstrings"),
    ("T78", "t78"),
    ("Thomas Schumacher", "thomas_schumacher"),
    ("Tiësto", "tiesto"),
    ("Tiga", "tiga"),
    ("Tinlicker", "tinlicker"),
    ("Toman", "toman"),
    ("Trace", "trace"),
    ("Underworld", "underworld"),
    ("Vieze Asbak", "vieze_asbak"),
    ("Vintage Culture", "vintage_culture"),
    ("Viperactive", "viperactive"),
    ("Virtual Riot", "virtual_riot"),
    ("VTSS", "vtss"),
    ("Walker & Royce", "walker_and_royce"),
    ("VNSSA", "vnssa"),
    ("Warface", "warface"),
    ("Warung", "warung"),
    ("Wax Motif", "wax_motif"),
    ("Westend", "westend"),
    ("Whethan", "whethan"),
    ("William Black", "william_black"),
    ("Wooli", "wooli"),
    ("YDG", "ydg"),
    ("Yosuf", "yosuf"),
]

if __name__ == "__main__":
    initialize_token()
    os.makedirs("data", exist_ok=True)

    total = len(EDC_2026_ARTISTS)
    collected = 0
    skipped = 0
    failed = 0
    failed_artists = []

    for i, (name, filename) in enumerate(EDC_2026_ARTISTS):
        output_path = f"data/{filename}.csv"

        if os.path.exists(output_path):
            print(f"[{i+1}/{total}] Skipping {name} (already exists)")
            skipped += 1
            continue

        print(f"\n{'='*60}")
        print(f"[{i+1}/{total}] Collecting: {name}")
        print(f"{'='*60}")

        try:
            collect_artist(name, output_path)
            collected += 1
        except ValueError as e:
            print(f"  SKIPPED: {e}")
            failed += 1
            failed_artists.append((name, str(e)))
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1
            failed_artists.append((name, str(e)))

        time.sleep(1)

    print(f"\n{'='*60}")
    print(f"COLLECTION COMPLETE")
    print(f"{'='*60}")
    print(f"Collected: {collected}")
    print(f"Skipped (already existed): {skipped}")
    print(f"Failed: {failed}")

    if failed_artists:
        print(f"\nFailed artists:")
        for name, error in failed_artists:
            print(f"  - {name}: {error}")
