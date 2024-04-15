#################################### Agenda ###################################################
## DealScan Leadbank
## TFIDF Measures for Banks - Auto & Manual
## Compustat
## FR Y9-C

# Define the base working directory
wd = "/shared/"

# Define specific paths based on the base directory
data = f"{wd}/Data"
raw = f"{data}/Raw"
processed = f"{data}/Processed"
codes = f"{wd}/Codes"

#=====# 1. Import Necessary Package #=====#
import wrds
import pandas as pd
import numpy as np
import fuzzy_match
db = wrds.Connection()

# Define start and end dates
startdate = '2010-01-01'
enddate = '2023-12-31'

#################################### PART I - DealScan Leadbank ####################################
#=====# Read in DealScan from WRDS, for our sample period 2010 - 2023 #=====#
dealscan_query = f"""SELECT * FROM dealscan.dealscan WHERE tranche_active_date >= '{startdate}' AND tranche_active_date <= '{enddate}';"""
dealscan = db.raw_sql(dealscan_query)

#=====# and create the lead bank, syndicated loans sub-samaple #=====#
## duplicates drop from the original dataset
dealscan_leadbank = dealscan.drop_duplicates()

## Syndicate loans Only
dealscan_leadbank = dealscan_leadbank[dealscan_leadbank['distribution_method'] == "Syndication"]

## Lead Banks Only
dealscan_leadbank['lead'] = np.where(dealscan_leadbank['primary_role'].isin(["Lead arranger", "Lead bank", "Lead underwriter", "Lead participant", "Lead Left", "Lead manager", "Lead underwriter"]), 1, 0)
dealscan_leadbank = dealscan_leadbank[dealscan_leadbank['lead'] == 1]
dealscan_leadbank = dealscan_leadbank[dealscan_leadbank['tranche_o_a'] == "Origination"]

## Columns to keep from the full datset
# columns_to_keep = [
#     'lender_parent_name', 'lender_parent_id',
#     'lender_name', 'lender_id', 'lender_commit', 'lender_share',
#     'lender_operating_country', 'lender_parent_operating_country',
#     'borrower_name', 'borrower_id', 'ticker', 'perm_id',
#     'city', 'state_province', 'zip', 'country', 'region', 'sales_size',
#     'broad_industry_group', 'major_industry_group', 'sic_code', 'naic',
#     'parent', 'parent_ticker', 'senior_debt_to_ebitda', 'total_debt_to_ebitda',
#     'number_of_lenders', 'lender_region', 'lender_parent_region', 'lpc_deal_id',
#     'deal_permid', 'deal_amount', 'deal_amount_converted', 'deal_active_date', 'deal_currency',
#     'lpc_tranche_id', 'tranche_permid', 'tranche_active_date',
#     'tranche_maturity_date', 'tranche_amount',
#     'league_table_amount', 'league_table_amount_converted', 'tranche_amount_converted',
#     'tranche_currency', 'market_segment', 'market_of_syndication', 'country_of_syndication',
#     'league_table_credit', 'league_table_tranche_date'
# ]

columns_to_keep = [
    'lender_parent_name', 'lender_parent_id',
    'lender_name', 'lender_id', 'lender_share',
    'lender_operating_country', 'lender_parent_operating_country',
    'borrower_name', 'borrower_id', 'ticker', 'perm_id',
    'city', 'state_province', 'zip', 'country', 'region', 
    'lpc_tranche_id', 'lpc_deal_id', 'tranche_active_date',
    'broad_industry_group', 'major_industry_group', 'sic_code', 'naic',
    'deal_permid', 'deal_amount', 'deal_amount_converted', 'deal_active_date', 'deal_currency',
    'tranche_maturity_date', 'tranche_amount', 'tranche_amount_converted','tranche_currency'
]

dealscan_leadbank = dealscan_leadbank[columns_to_keep]
dealscan_leadbank = dealscan_leadbank.drop_duplicates()

# Datetime formatting and creating year variables
dealscan_leadbank['deal_active_date'] = pd.to_datetime(dealscan_leadbank['deal_active_date'])
dealscan_leadbank['year'] = dealscan_leadbank['deal_active_date'].dt.year
dealscan_leadbank['month'] = dealscan_leadbank['deal_active_date'].dt.month
dealscan_leadbank['quarter'] = dealscan_leadbank['deal_active_date'].dt.quarter

# Sort by the following variables to keep the non-missing rows at last
dealscan_leadbank.sort_values(by=['broad_industry_group', 'major_industry_group', 'sic_code', 'naic'], inplace=True, ascending=False)

# Drop duplicates with specific columns considered
dealscan_leadbank.drop_duplicates(subset=['lender_id', 'borrower_id', 'lpc_tranche_id', 'lpc_deal_id', 'deal_active_date'], inplace=True, ignore_index=True)

# Sort again by date before dropping
dealscan_leadbank.sort_values(by='deal_active_date', inplace=True)

# Drop duplicates again to keep the earliest ones
dealscan_leadbank.drop_duplicates(subset=['lender_id', 'borrower_id', 'lpc_tranche_id', 'lpc_deal_id'], inplace=True, ignore_index=True)

#=====# Final Clean-up #=====#
# Keep only lead_bank with valid lender_parent_id
dealscan_leadbank = dealscan_leadbank.loc[dealscan_leadbank['lender_parent_id'] != "N/A"]
dealscan_leadbank['lender_parent_id'] = pd.to_numeric(dealscan_leadbank['lender_parent_id'])

# List of specified column names to prefix
columns_to_prefix_for_borrower = ['ticker', 'perm_id', 'city', 'state_province', 'zip', 'country', 'region', 
                     'broad_industry_group', 'major_industry_group', 'sic_code', 'naic']

# Add the prefix "borrower" to each specified column name
new_borrower_column_names = {column: f'borrower_{column}' for column in columns_to_prefix_for_borrower if column in dealscan_leadbank.columns}

# Rename the columns in the DataFrame
dealscan_leadbank.rename(columns=new_borrower_column_names, inplace=True)

#################################### PART I.I - DealScan Leadbank Cross Walk ####################################
#=====# Import Legacy to New Dealscan Cross Walk, keep only obs with nonmissing id #=====# 
lpc_loanconnector_company_id_map = db.raw_sql("SELECT * FROM dealscan.lpc_loanconnector_company_id_map;")
lpc_loanconnector_company_id_map = lpc_loanconnector_company_id_map.loc[~lpc_loanconnector_company_id_map['lpc_company_id'].isna()]

#=====# Import Legacy Dealscan to Compustat Cross Walk, keep only obs with nonmissing id #=====# 
dealscan_compustat_crosswalk = pd.read_excel(fr"{raw}/DealScan/ds-cs-link-post-202401.xlsx", sheet_name="links", header=0)
## Cleaning
dealscan_compustat_crosswalk.loc[(dealscan_compustat_crosswalk['ds_ticker'].isna())| (dealscan_compustat_crosswalk['cs_ticker'].isna()), 'score_ticker_match']= np.nan
## Duplicates drop on gvkey borrowercompanyid, ignoring facilityid
dealscan_compustat_crosswalk = dealscan_compustat_crosswalk.drop_duplicates(subset=['gvkey', 'borrowercompanyid'], keep='first')
print(len(dealscan_compustat_crosswalk)) ##35784
# Sort the DataFrame
dealscan_compustat_crosswalk = dealscan_compustat_crosswalk.sort_values(by=['borrowercompanyid', 'score_company_match', 'score_ticker_match', 'confidence_score'], 
                    ascending=[True, False, False, False])
dealscan_compustat_crosswalk = dealscan_compustat_crosswalk.drop_duplicates(subset=['borrowercompanyid'], keep='first')
print(len(dealscan_compustat_crosswalk)) ##34716

## Inner join, akin to 1;1 merge in Stata, with keeping on _merge == 3
dealscan_compustat_crosswalk_updated = pd.merge(dealscan_compustat_crosswalk, lpc_loanconnector_company_id_map, left_on = 'borrowercompanyid', right_on = 'lpc_company_id', how = 'outer', indicator=True)
# _merge
# right_only    112485
# both           34652
# left_only         64
# Name: count, dtype: int64

#################################### PART II - Compustat from WRDS ####################################
#=====# Read in Compustat #=====#
compustat_query = f"""
SELECT dt, gvkey, conm, tic, cik, exchg, datadate
FROM comp_na_daily_all.funda
WHERE datadate BETWEEN '{startdate}' AND '{enddate}'
AND DATAFMT='STD' 
AND INDFMT='INDL'
AND CONSOL='C' 
AND POPSRC='D'
AND FYEAR IS NOT NULL 
AND FYR != 0
"""
compustat = db.raw_sql(compustat_query, date_cols=['datadate'])
compustat['year'] = compustat['datadate'].dt.year


compustat.sort_values(by=['gvkey', 'dt'], inplace = True, ascending=False)
compustat.drop_duplicates(subset=['gvkey', 'dt'], inplace=True, ignore_index=True) ## about 101 company-year having 1 duplicates, magnitudes small so dropped

compustat['gvkey'] = pd.to_numeric(compustat['gvkey'])

#=====# Merge compustat with Roberts Crosswalk to get the dealscan ID
compustat_with_dealscan_id = pd.merge(compustat, dealscan_compustat_crosswalk_updated, on = 'gvkey', how = 'inner')
# _merge
# both          102584
# left_only      43622
# right_only     22855
compustat_with_dealscan_id = compustat_with_dealscan_id[['dt', 'gvkey', 'conm', 'tic', 'cik', 'exchg', 'datadate', 'year', 'loanconnector_company_id']]

dealscan_leadbank_with_compustat = pd.merge(dealscan_leadbank, compustat_with_dealscan_id, left_on = ['year', 'borrower_id'], right_on = ['year', 'loanconnector_company_id'], how = 'left', indicator = True)
dealscan_leadbank_with_compustat['_merge'].value_counts()
# _merge
# right_only    102328
# left_only      25853
# both            1409
dealscan_leadbank_with_compustat.rename(columns={'_merge': 'compustat_merge'}, inplace=True)

#################################### PART III. Call Report ####################################
call_report_bhck_query = f"""SELECT rssd9001, rssd9999, rssd9017, bhck2170 FROM bank_all.wrds_holding_bhck_2
WHERE rssd9999 >= '{startdate}' AND rssd9999 <= '{enddate}';"""
call_report_bhck = db.raw_sql(call_report_bhck_query)
call_report_bhck['rssd9999'] = pd.to_datetime(call_report_bhck['rssd9999'])
call_report_bhck['year'] = call_report_bhck['rssd9999'].dt.year
call_report_bhck['quarter'] = call_report_bhck['rssd9999'].dt.quarter

## Annualized by caluclating the mean across 4 quarters (majority of the companies report once a year)
call_report_bhck_a = call_report_bhck.groupby(['rssd9001', 'rssd9017', 'year'])['bhck2170'].mean().reset_index()

## About 240 duplicates (some due to names differences, some due to rssd ownership changes)
## for now we will just do a duplicates drop
call_report_bhck_a.sort_values(by=['rssd9001', 'bhck2170'], ascending=False)

call_report_bhck_a.sort_values(by=['rssd9001', 'year', 'bhck2170'], ascending=False, inplace=True)
call_report_bhck_a.drop_duplicates(subset=['rssd9001', 'year'], inplace=True, ignore_index=True) ## this drop favors bigger companies if there were a change of rssd ownership and both comapnies have non-missing total assets

## Fuzzy matching with names from FFIEC call report
link_bhck = fuzzy_match.gen_match_links(dealscan_leadbank, 'lender_parent_name', 'lender_parent_id', call_report_bhck_a, 'rssd9017', 'rssd9001', compare_initials=True, conf_threshold=0.6)

call_report_bhck_a_matched = pd.merge(call_report_bhck_a, link_bhck[['lender_parent_id', 'rssd9001', 'year']], on = ['rssd9001', 'year'], how = 'right')

dealscan_leadbank_with_compustat_callreport = pd.merge(dealscan_leadbank_with_compustat, call_report_bhck_a_matched, on = ['lender_parent_id', 'year'], how = 'outer', indicator = True)
dealscan_leadbank_with_compustat_callreport['_merge'].value_counts()
dealscan_leadbank_with_compustat_callreport.rename(columns={'_merge': 'callreport_merge'}, inplace=True)

################################################### TFIDF Measures Results ###################################################
tfidf_autokeywords_scores = pd.read_csv(f"{processed}/10K_NLP/panel_tf_auto_full_data.csv")
tfidf_autokeywords_scores.rename(columns={'Frequency': 'measure_auto', 'Count': 'Count_Auto'}, inplace=True)

tfidf_manualkeywords_scores = pd.read_csv(f"{processed}/10K_NLP/panel_tf_manual_full_data.csv")
tfidf_manualkeywords_scores.rename(columns={'Frequency': 'measure_manual', 'Count': 'Count_Manual'}, inplace=True)

tfidf_frequncy_scores = pd.merge(tfidf_autokeywords_scores, tfidf_manualkeywords_scores, on = ['Company', 'Year'])

## Bring in CIK for each compnay since tickers are not universal across the globe
url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
sp500_table = pd.read_html(url)[0]
tfidf_frequncy_scores = pd.merge(tfidf_frequncy_scores, sp500_table[['Symbol', 'CIK']], left_on = 'Company', right_on = 'Symbol', how = 'left')

dealscan_leadbank_with_compustat_callreport['cik'] = pd.to_numeric(dealscan_leadbank_with_compustat_callreport['cik'])

tfidf_frequncy_scores = tfidf_frequncy_scores.groupby(['Year', 'CIK']).agg({
    'measure_auto': 'mean',
    'measure_manual': 'mean'
}).reset_index()

# Merge DealScan with TFIDF measures
dealscan_leadbank_with_compustat_callreport_tfidf = pd.merge(dealscan_leadbank_with_compustat_callreport, tfidf_frequncy_scores, left_on = ['year', 'cik'], right_on = ['Year', 'CIK'], how = 'left', indicator=True)
# _merge
# left_only     27207
# both             55
# right_only        0
pd.merge(dealscan_leadbank_with_compustat_callreport, tfidf_frequncy_scores, left_on = ['year', 'cik'], right_on = ['Year', 'CIK'], how = 'left', indicator=True)
dealscan_leadbank_with_compustat_callreport_tfidf.rename(columns={'_merge': 'tfidf_measure_merge'}, inplace=True)

################################################### Clean Up ###################################################
#=====# Renaming #=====#
# List of specified column names to prefix
columns_to_prefix_ds = ['lender_parent_name', 'lender_parent_id', 'lender_name', 'lender_id', 'lender_share', 
                        'lender_operating_country', 'lender_parent_operating_country', 'borrower_name', 
                        'borrower_id', 'borrower_ticker', 'borrower_perm_id', 'borrower_city', 
                        'borrower_state_province', 'borrower_zip', 'borrower_country', 'borrower_region', 
                        'lpc_tranche_id', 'lpc_deal_id', 'tranche_active_date', 'borrower_broad_industry_group', 
                        'borrower_major_industry_group', 'borrower_sic_code', 'borrower_naic', 'deal_permid', 
                        'deal_amount', 'deal_amount_converted', 'deal_active_date', 'deal_currency', 
                        'tranche_maturity_date', 'tranche_amount', 'tranche_amount_converted', 'tranche_currency']

# Add the prefix "ds_" to each specified column name
new_column_names_ds = {column: f'ds_{column}' for column in columns_to_prefix_ds if column in dealscan_leadbank_with_compustat_callreport_tfidf.columns}

# Rename the columns in the DataFrame
dealscan_leadbank_with_compustat_callreport_tfidf.rename(columns=new_column_names_ds, inplace=True)

################### Renaming and Selecting #######################
# List of specified column names to prefix
columns_to_prefix_cs = ['dt', 'gvkey', 'conm', 'tic', 'cik', 'exchg', 'datadate']

# Add the prefix "cs_" to each specified column name
new_column_names_cs = {column: f'cs_{column}' for column in columns_to_prefix_cs if column in dealscan_leadbank_with_compustat_callreport_tfidf.columns}

# Rename the columns in the DataFrame
dealscan_leadbank_with_compustat_callreport_tfidf.rename(columns=new_column_names_cs, inplace=True)

cr_rename_columns = {
    'rssd9017': 'cr_bhc_name',
    'bhck2170': 'cr_bhc_total_assets'
}

# Rename the columns in the DataFrame
dealscan_leadbank_with_compustat_callreport_tfidf.rename(columns=cr_rename_columns, inplace=True)

# List of specified column names to prefix

columns_to_prefix_tfidf = ['measure_auto', 'measure_manual', 'Symbol', 'CIK']

# Add the prefix "tfidf_" to each specified column name
new_column_names_tfidf = {column: f'tfidf_{column}' for column in columns_to_prefix_tfidf if column in dealscan_leadbank_with_compustat_callreport_tfidf.columns}

# Rename the columns in the DataFrame
dealscan_leadbank_with_compustat_callreport_tfidf.rename(columns=new_column_names_tfidf, inplace=True)

# List of columns to drop
columns_to_drop = ['Year', 'loanconnector_company_id']

# Drop the specified columns from the DataFrame
dealscan_leadbank_with_compustat_callreport_tfidf.drop(columns=columns_to_drop, inplace=True)

dealscan_leadbank_with_compustat_callreport_tfidf.to_csv(f'{processed}/dealscan_complete.csv', index=False)