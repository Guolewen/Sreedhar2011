import pandas as pd
from collections import Counter
import time
import numpy as np


facility = pd.read_csv(r'D:\wooldride_econometrics\paper_liq_replicating\Sreedhar\data\facility.csv')
facilitydates = pd.read_csv(r'D:\wooldride_econometrics\paper_liq_replicating\Sreedhar\data\facilitydates.csv')
facilitysponsor = pd.read_csv(r'D:\wooldride_econometrics\paper_liq_replicating\Sreedhar\data\facilitysponsor.csv')
lendershares = pd.read_csv(r'D:\wooldride_econometrics\paper_liq_replicating\Sreedhar\data\lendershares.csv')
company = pd.read_csv(r'D:\wooldride_econometrics\paper_liq_replicating\Sreedhar\data\company.csv')
currfacpricing = pd.read_csv(r'D:\wooldride_econometrics\paper_liq_replicating\Sreedhar\data\currfacpricing.csv')
performancepricing = pd.read_csv(r'D:\wooldride_econometrics\paper_liq_replicating\Sreedhar\data\performancepricing.csv')

def classify_lead_bank(x, **kwargs):
    if x['LenderRole'] in kwargs['lendroles'] and x['BankAllocation'] > 25:
        return 1
    elif x['LenderRole'] == 'Sole Lender':
        return 1
    elif x['LeadArrangerCredit'] == 'Yes':
        return 1
    else:
        return 0

def deal_fees(currfacpricing):
    def upfront_fee(x):
        queried_upfront = x[x['Fee'] == 'Upfront Fee']
        queried_annual = x[x['Fee'] == 'Annual Fee']
        if queried_upfront.empty and queried_annual.empty:
            return [np.NaN, np.NaN]
        elif len(queried_upfront) == 1 and queried_annual.empty:
            return [np.float64(queried_upfront['MaxBps']), np.NaN]
        elif len(queried_annual) == 1 and queried_upfront.empty:
            return [np.NaN, np.float64(queried_annual['MaxBps'])]
        elif len(queried_upfront) == 1 and len(queried_annual) == 1:
            return [np.float64(queried_upfront['MaxBps']), np.float64(queried_annual['MaxBps'])]
        else:
            return None

    drop_dup = currfacpricing.drop_duplicates(subset='FacilityID').reset_index(drop=True)
    drop_dup[['UpfrontFee', 'AnnualFee']] = pd.DataFrame(currfacpricing.groupby(['FacilityID']).apply(upfront_fee).to_list())

    return drop_dup


def assign_lead_bank(lendershares):
    lendershares['LeadBankIndicator'] = lendershares[['LenderRole', 'BankAllocation', 'LeadArrangerCredit']].apply(classify_lead_bank, axis=1,
                                                                          lendroles=['Agent', 'Admin agent', 'Arranger'
                                                                                     , 'Lead bank'])
    return lendershares

def merge_facility_with_lendshares(facility, lendershares, fee_df):
    merged_df = facility.merge(lendershares[['FacilityID', 'LenderRole', 'BankAllocation', 'Lender',
                                             'LeadArrangerCredit', 'LeadBankIndicator']], how='left', on='FacilityID')
    merged_df = fee_df[['FacilityID', 'AllInDrawn', 'AllInUndrawn', 'UpfrontFee', 'AnnualFee']].merge(merged_df, how='left', on="FacilityID")
    usa = merged_df[(merged_df['CountryOfSyndication'] == 'USA') & (merged_df['LeadBankIndicator']==1)]
    timeperiod = usa[(usa['FacilityStartDate'] > 19860100) & (usa['FacilityStartDate'] < 20040100)].reset_index(drop=True)
    # merge the company and delete firms with SICCode starting with 6
    comp_merged = timeperiod.merge(company[['CompanyID', 'PrimarySICCode']], how='left', left_on='BorrowerCompanyID',
                                                                             right_on='CompanyID')
    sample = comp_merged[(comp_merged['PrimarySICCode'] < 6000) | (comp_merged['PrimarySICCode'] >= 7000)].reset_index(drop=True)
    return sample

def calculate_relation_variables(sample):

    def calculate_rels(x, **kwargs):
        print('f id', x['FacilityID'])
        borrower_id = x['BorrowerCompanyID']
        lender = x['Lender']
        begin = x['FacilityStartDate'] - 50000
        end = x['FacilityStartDate'] - 1
        sample = kwargs['sample']
        # query through the whole sample to find out the borrowers' previous 5 year records
        queried_sample = sample[(sample['BorrowerCompanyID'] == borrower_id) & (sample['FacilityStartDate'] >= begin) &
                                (sample['FacilityStartDate'] <= end)]
        if queried_sample.empty:
            return [0, 0, 0]
        else:
            same_lender = queried_sample[queried_sample['Lender'] == lender]
            if same_lender.empty:
                return [0, 0, 0]
            else:
                rel_amount = same_lender['FacilityAmt'].sum() / queried_sample['FacilityAmt'].sum()
                rel_number = len(same_lender) / len(queried_sample)
                return [1, rel_amount, rel_number]


    sample[['REL_Dummy', 'REL_Amount', 'REL_Number']] = sample.apply(calculate_rels, axis=1, result_type='expand', sample=sample)
    final = sample.drop_duplicates(subset=['FacilityID']).reset_index(drop=True)
    # assign the highest values to REL variables
    final['REL_Dummy'] = sample.groupby(['FacilityID'])['REL_Dummy'].max().reset_index(drop=True)
    final['REL_Amount'] = sample.groupby(['FacilityID'])['REL_Amount'].max().reset_index(drop=True)
    final['REL_Number'] = sample.groupby(['FacilityID'])['REL_Number'].max().reset_index(drop=True)
    return final


def panel_a(final):
    final['Year'] = final['FacilityStartDate'].apply(lambda x: str(x)[0:4])
    panel_a = final.groupby(['Year', 'REL_Dummy']).size().unstack(level=-1)
    panel_a['Total'] = panel_a[0.0] + panel_a[1.0]
    panel_a = panel_a.append(panel_a.sum().rename("Total").to_frame().transpose())
    return panel_a

def panel_b(final):
    def extractsic(x):
        number = int(x)
        if number < 1000:
            return '0'
        else:
            return str(number)[0]

    final['FirstSCICode'] = final['PrimarySICCode'].apply(extractsic)
    panel_b = final.groupby(['FirstSCICode', 'REL_Dummy']).size().unstack(level=-1)
    panel_b['Total'] = panel_b[0.0] + panel_b[1.0]
    panel_b = panel_b.append(panel_b.sum().rename("Total").to_frame().transpose())
    return panel_b
    
def panel_c(final):
    panel_c = final.groupby(['PrimaryPurpose', 'REL_Dummy']).size().unstack(level=-1)
    panel_c = panel_c.fillna(0)
    panel_c['Total'] = panel_c[0.0] + panel_c[1.0]
    panel_c = panel_c.append(panel_c.sum().rename("Total").to_frame().transpose())
    return panel_c


if __name__ == '__main__':
    t1 = time.time()
    fees = deal_fees(currfacpricing)
    assigned_df = assign_lead_bank(lendershares)
    sample = merge_facility_with_lendshares(facility, assigned_df, fees)
    final = calculate_relation_variables(sample)
    print(panel_a(final))
    print(panel_b(final))
    print(panel_c(final))
    t2 = time.time()
    print("Total Runtime is", t2 - t1)

"""
Panel A
REL_Dummy    0.0    1.0  Total
1986          96      2     98
1987         805     53    858
1988        1678    254   1932
1989        1488    384   1872
1990        1170    476   1646
1991        1072    473   1545
1992        1235    545   1780
1993        1551    665   2216
1994        1780    949   2729
1995        1690    967   2657
1996        2550   1096   3646
1997        3414   1421   4835
1998        2661   1091   3752
1999        3201   1082   4283
2000        2846   1464   4310
2001        2475   1226   3701
2002        2653   1107   3760
2003        2729   1323   4052
Total      35094  14578  49672

Panel B
REL_Dummy    0.0    1.0  Total
0            193     69    262
1           2516   1073   3589
2           5745   2489   8234
3           9414   3633  13047
4           5132   2587   7719
5           5385   2124   7509
7           4290   1633   5923
8           2352    960   3312
9             67     10     77
Total      35094  14578  49672

Panel C
REL_Dummy                   0.0      1.0    Total
Acquis. line             2172.0    695.0   2867.0
CP backup                1093.0   1386.0   2479.0
Capital expend.            84.0     22.0    106.0
Corp. purposes           9732.0   3723.0  13455.0
Cred Enhanc                42.0     27.0     69.0
Debt Repay.              6856.0   3723.0  10579.0
Debtor-in-poss.           269.0    187.0    456.0
ESOP                       84.0     21.0    105.0
Equip. Purch.             170.0     45.0    215.0
Exit financing             29.0     24.0     53.0
IPO Relat. Finan.          52.0     12.0     64.0
LBO                      2565.0    384.0   2949.0
Lease finance              13.0      4.0     17.0
Mort. Warehse.              6.0      1.0      7.0
Other                     413.0    200.0    613.0
Proj. finance             508.0     92.0    600.0
Purch. Hardware             1.0      1.0      2.0
Purch. Software/Servs.      1.0      0.0      1.0
Real estate               184.0     50.0    234.0
Rec. Prog.                 20.0      7.0     27.0
Recap.                   1069.0    347.0   1416.0
Securities Purchase        61.0     15.0     76.0
Ship finance                2.0      0.0      2.0
Spinoff                   291.0     48.0    339.0
Stock buyback             215.0     64.0    279.0
Takeover                 3513.0   1636.0   5149.0
TelcomBuildout             56.0     28.0     84.0
Trade finance              40.0     12.0     52.0
Undisclosed                 3.0      0.0      3.0
Work. cap.               5550.0   1824.0   7374.0
Total                   35094.0  14578.0  49672.0
Total Runtime is 250.94556164741516


"""