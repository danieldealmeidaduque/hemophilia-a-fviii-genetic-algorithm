import re
import pandas as pd
from numpy import NaN
from Bio.Data.IUPACData import protein_letters_1to3_extended as prot1to3

from auxiliar import exception_handler, strip_string, sev2int


# ------------------------ Read Input Files -------------------------------------- #


@ exception_handler
def read_aa_dm_file(input_path):
    '''read amino acids distance matrix input file to a dataframe'''
    df = pd.read_excel(input_path, index_col=0)
    # remove extra spaces from string values
    df = df.applymap(lambda s: strip_string(s))
    # padronize column to aa capitalized
    df.columns = [c.capitalize() for c in df.columns]
    # padronize index to aa capitalized
    df.index = [c.capitalize() for c in df.index]
    # sort index alphabetically
    df = df.sort_index(key=lambda x: x)
    # sort columns alphabetically
    columns = sorted(df.columns.tolist())
    df = df[columns]

    return df


@ exception_handler
def read_rsa_file(input_path, csv_sep='\t'):
    '''read and clean the rsa input file to a dataframe'''
    df = pd.read_csv(input_path, sep=csv_sep)
    # remove extra spaces from string values
    df = df.applymap(lambda s: strip_string(s))
    # padronize column names to access with dot
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    # eliminate rows with NaN value
    df = df.dropna()
    # convert pos_hgvs column to int to use as index
    df = df.astype({'pos_hgvs': int})
    # set pos_hgvs as index
    df = df.set_index('pos_hgvs')
    # change values of the residue to 3 character aminoacid
    df.residue = df.residue.apply(lambda p: prot1to3[p])

    return df


@ exception_handler
def read_pm_file(input_path, csv_sep='\t'):
    '''read and clean the point mutations input file to a dataframe'''
    df = pd.read_csv(input_path, sep=csv_sep)
    # remove extra spaces from string values
    df = df.applymap(lambda s: strip_string(s))
    # padronize column names to access with dot
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    # set case_id as index
    df = df.set_index('case_id')

    # func to clean protein_change column to clear view
    def clean_prot_change(prot_change):
        pattern = '(?<=\()(.*?)(?=\))'
        clean_str = re.search(pattern, prot_change)
        clean_str = clean_str.group().replace(' ', '')
        clean_str = clean_str.strip()
        return clean_str

    # func to get only the wild amino acid from protein_change column
    def get_wild_aa(prot_change):
        return prot_change[:3]

    # func to get only the new amino acid from protein_change column
    def get_new_aa(row):
        effect = row.effect.strip()
        if effect == 'Missense':
            return row.protein_change[-3:]
        else:
            return '---'

    # apply changes in the dataframe
    df.protein_change = df.protein_change.apply(clean_prot_change)
    df['wild_aa'] = df.protein_change.apply(get_wild_aa)
    df['new_aa'] = df.apply(get_new_aa, axis=1)

    return df

# ------------------------ Insert in dataframe -------------------------------------- #


@ exception_handler
def insert_dist_aa(df, aa_dm_df):
    '''function to insert distance between two amino acids in each row of the dataframe'''
    def get_dist_aa(row):
        '''get distance between two amino acids'''
        wild = row.wild_aa
        new = row.new_aa

        if new != '---':
            return aa_dm_df[wild][new]
        else:
            return NaN

    df['dist_aa'] = df.apply(get_dist_aa, axis=1)


@ exception_handler
def insert_rsa(df, rsa_df):
    '''function to insert rsa value based on position in each row of the dataframe'''
    def cross_check_residue_is_wild_aa(row):
        '''get rsa after - cross check if residue in rsa is the correct wild amino acid'''
        pos = row.position_hgvs
        wild = row.wild_aa

        # check if point mutation position is in the rsa matrix
        if pos in rsa_df.index:
            residue = rsa_df.loc[pos].residue
            rsa = rsa_df.loc[pos].rsa

            # check if residue in rsa matrix is equal to the wild amino acid in the dataframe
            is_same_aa = wild == residue

            # return rsa if residue is equal to the wild amino acid
            return rsa if is_same_aa else NaN

        else:
            # rsa matrix don't have specified position
            return NaN

    df['rsa'] = df.apply(cross_check_residue_is_wild_aa, axis=1)

# ------------------------ Clean Data -------------------------------------- #


@ exception_handler
def get_initial_df(aa_dm_path, rsa_path, pm_path):
    '''function to read all files and put all informations in a singles dataframe'''

    df_input_pm = read_pm_file(pm_path, csv_sep='\t')
    # print_highlighted('Input: Point mutations')
    # print(df_input_pm)

    df_input_aa_dm = read_aa_dm_file(aa_dm_path)
    insert_dist_aa(df_input_pm, df_input_aa_dm)
    # print_highlighted('Input: Amino acids distance matrix')
    # print(df_input_aa_dm)

    df_input_rsa = read_rsa_file(rsa_path, csv_sep='\t')
    insert_rsa(df_input_pm, df_input_rsa)
    # print_highlighted('Input: Relative surface area')
    # print(df_input_rsa)

    df_input_pm.dropna(inplace=True)
    # print_highlighted('Dataframe without NaN values')
    # print(df_input_pm)

    df_input_pm.sort_values(by='position_hgvs', inplace=True)

    return df_input_pm

# ------------------------ GA Auxiliar ------------------------------------------- #


@ exception_handler
def filter_unique_mutations(df):
    '''function filter the dataframe based on same mutations'''

    df_aux = df.copy()
    df_aux['sev'] = df_aux['severity'].apply(lambda s: sev2int[s])

    df_aux = df_aux[df_aux['effect'] == 'Missense']

    df_grouped = df_aux.groupby(['wild_aa', 'new_aa'])

    g_pos = df_grouped['position_hgvs'].apply(lambda x: list(x.values))
    g_rsa = df_grouped['rsa'].apply(lambda x: list(x.values))
    g_sev = df_grouped['sev'].apply(lambda x: list(x.values))

    df_aux['position_hgvs'] = df_aux.apply(
        lambda r: g_pos[r.wild_aa][r.new_aa], axis=1)

    df_aux['rsa'] = df_aux.apply(
        lambda r: g_rsa[r.wild_aa][r.new_aa], axis=1)

    df_aux['sev'] = df_aux.apply(
        lambda r: g_sev[r.wild_aa][r.new_aa], axis=1)

    df_aux.drop_duplicates(subset=['wild_aa', 'new_aa'], inplace=True)

    print(df_aux)
    return df_aux
