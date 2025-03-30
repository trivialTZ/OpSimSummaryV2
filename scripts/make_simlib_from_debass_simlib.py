import argparse
import os
import re
import sqlite3

def read_debass_simlib(simlib_filename):


    lib_list = []
    with open(simlib_filename, 'r') as f:
        lines = f.readlines()

    current_lib = None
    reading_lib = False

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Skip documentation headers
        if line.startswith("DOCUMENTATION:") or line.startswith("DOCUMENTATION_END:") or line.startswith("BEGIN LIBGEN"):
            continue

        # Check for start of LIBID block
        if line.startswith("LIBID:"):
            current_lib = {'header': {}, 'observations': []}
            reading_lib = True
            # Extract LIBID value, e.g., "LIBID:  1362267"
            parts = line.split()
            if len(parts) >= 2:
                current_lib['header']['LIBID'] = parts[1]
            continue

        # Check for end of LIBID block
        if line.startswith("END_LIBID:"):
            if current_lib is not None:
                lib_list.append(current_lib)
            current_lib = None
            reading_lib = False
            continue

        # Process header lines (non-observation entries)
        if reading_lib and not line.startswith("S:"):
            pairs = re.findall(r'(\w+):\s*(.+?)(?=\s+\w+:|$)', line)
            for key, value in pairs:
                current_lib['header'][key.strip()] = value.strip()
            continue


        # Process observation entries starting with "S:"
        if reading_lib and line.startswith("S:"):
            obs_line = line[2:].strip()
            obs_parts = obs_line.split()
            if len(obs_parts) >= 12:
                try:
                    obs = {
                        'MJD': float(obs_parts[0]),
                        'IDEXPT': int(obs_parts[1]),
                        'BAND': obs_parts[2],
                        'GAIN': float(obs_parts[3]),
                        'RDNOISE': float(obs_parts[4]),
                        'SKYSIG': float(obs_parts[5]),
                        'PSF1': float(obs_parts[6]),
                        'PSF2': float(obs_parts[7]),
                        'PSFRAT': float(obs_parts[8]),
                        'ZP': float(obs_parts[9]),
                        'ZPERR': float(obs_parts[10]),
                        'MAG': float(obs_parts[11]),
                    }
                except ValueError:
                    # Skip malformed observation lines
                    continue
                current_lib['observations'].append(obs)
    return lib_list


def create_db_from_simlib(lib_data, db_file):
    """
    Create SQLite database from parsed SIMLIB data.

    Creates two tables:
    1. lib_header: Metadata for each LIBID (RA, DEC, MWEBV, NOBS, etc.)
    2. observations: Observation records linked to LIBID
    """
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()

    # Create header table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS lib_header (
            LIBID TEXT PRIMARY KEY,
            RA TEXT,
            DEC TEXT,
            MWEBV TEXT,
            NOBS TEXT,
            PIXSIZE TEXT,
            REDSHIFT TEXT,
            PEAKMJD TEXT,
            TEMPLATE_ZPT TEXT,
            TEMPLATE_SKYSIG TEXT
        )
    """)

    # Create observations table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            LIBID TEXT,
            MJD REAL,
            IDEXPT INTEGER,
            BAND TEXT,
            GAIN REAL,
            RDNOISE REAL,
            SKYSIG REAL,
            PSF1 REAL,
            PSF2 REAL,
            PSFRAT REAL,
            ZP REAL,
            ZPERR REAL,
            MAG REAL,
            FOREIGN KEY(LIBID) REFERENCES lib_header(LIBID)
        )
    """)

    # Insert data
    for lib in lib_data:
        header = lib['header']
        libid = header.get('LIBID', None)
        if not libid:
            continue

        # Insert header data
        cur.execute("""
            INSERT OR REPLACE INTO lib_header (LIBID, RA, DEC, MWEBV, NOBS, PIXSIZE, REDSHIFT, PEAKMJD, TEMPLATE_ZPT, TEMPLATE_SKYSIG)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            libid,
            header.get('RA'),
            header.get('DEC'),
            header.get('MWEBV'),
            header.get('NOBS'),
            header.get('PIXSIZE'),
            header.get('REDSHIFT'),
            header.get('PEAKMJD'),
            header.get('TEMPLATE_ZPT'),
            header.get('TEMPLATE_SKYSIG')
        ))

        # Insert observation records
        for obs in lib['observations']:
            cur.execute("""
                INSERT INTO observations (LIBID, MJD, IDEXPT, BAND, GAIN, RDNOISE, SKYSIG, PSF1, PSF2, PSFRAT, ZP, ZPERR, MAG)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                libid,
                obs['MJD'],
                obs['IDEXPT'],
                obs['BAND'],
                obs['GAIN'],
                obs['RDNOISE'],
                obs['SKYSIG'],
                obs['PSF1'],
                obs['PSF2'],
                obs['PSFRAT'],
                obs['ZP'],
                obs['ZPERR'],
                obs['MAG']
            ))
    conn.commit()
    conn.close()
    print(f"Database file created: {db_file}")


lib_data = read_debass_simlib("/Users/tz/PycharmProjects/OpSimSummaryV2/scripts/DEBASS.SIMLIB")
#create_db_from_simlib(lib_data, "/Users/tz/PycharmProjects/OpSimSummaryV2/scripts/debass_fake_new.db")

#if __name__ == "__main__":
#    main()
