import re
import pandas as pd
from numpy import NaN
from Bio.Data.IUPACData import protein_letters_1to3_extended as prot1to3

from auxiliar import exception_handler, strip_string


# ------------------------ Read Input Files -------------------------------------- #


@ exception_handler
def read_aa_dm_excel(input_path):
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
def read_rsa_csv(input_path, csv_sep='\t'):
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
def read_pm_csv(input_path, csv_sep='\t'):
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
def insert_dist_aa(df, df_dm):
    '''function to insert distance between two amino acids in each row of the dataframe'''
    def get_dist_aa(row):
        '''get distance between two amino acids'''
        wild = row.wild_aa
        new = row.new_aa

        if new != '---':
            return df_dm[wild][new]
        else:
            return NaN

    df['dist_aa'] = df.apply(get_dist_aa, axis=1)


@ exception_handler
def insert_rsa(df, df_rsa):
    '''function to insert rsa value based on position in each row of the dataframe'''
    def cross_check_residue_is_wild_aa(row):
        '''get rsa after - cross check if residue in rsa is the correct wild amino acid'''
        pos = row.position_hgvs
        wild = row.wild_aa

        # check if point mutation position is in the rsa matrix
        if pos in df_rsa.index:
            residue = df_rsa.loc[pos].residue
            rsa = df_rsa.loc[pos].rsa

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
def initial_df(pm_path, dm_path, rsa_path):
    '''function to read all files and put all informations in a singles dataframe'''
    df_pm = read_pm_csv(pm_path, csv_sep='\t')
    df = df_pm.copy()

    df_dm = read_aa_dm_excel(dm_path)
    insert_dist_aa(df, df_dm)

    df_rsa = read_rsa_csv(rsa_path, csv_sep='\t')
    insert_rsa(df, df_rsa)

    df.dropna(inplace=True)
    df.sort_values(by='position_hgvs', inplace=True)

    # print('\n\tOriginal Dataframe\n')
    # print(df.info())  # or df
    return df
