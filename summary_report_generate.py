import json
import ast, re
import math

import xlsxwriter
import pandas as pd
from openpyxl import load_workbook

# Estimate number of wrapped lines
def estimate_lines(text, col_width_chars):
    lines = text.split('\n')
    total_lines = 0
    for line in lines:
        total_lines += math.ceil(len(line) / col_width_chars) or 1
    return total_lines

def summary_report(product_name, input_data):
    file_path = '/home/ec2-user/kathiravan'
    workbook = xlsxwriter.Workbook(f"{file_path}/output/{product_name}_Summary.xlsx")
    
    worksheet = workbook.add_worksheet('Summary')
    worksheet.hide_gridlines(2)

    # Base font
    font_base = {'font_name': 'Aptos Narrow', 'font_size': 10}
    normal = workbook.add_format({**font_base, 'border': 1})
    wrap = workbook.add_format({**font_base, 'text_wrap': True})
    wrap_left = workbook.add_format({**font_base, 'text_wrap': True, 'align': 'left', 'valign': 'vcenter', 'border': 1})

    worksheet.set_column('B:B', 35, wrap)
    worksheet.set_column('C:I', 20, wrap)
    worksheet.set_column('J:J', 30)

    # Header formats 
    header_colored = workbook.add_format({
        **font_base,
        'align': 'center',
        'valign': 'vcenter',
        'bg_color': '#D9D9D9',
        'border': 1
    })

    header_wrap_colored = workbook.add_format({
        **font_base,
        'align': 'center',
        'valign': 'vcenter',
        'bg_color': '#D9D9D9',
        'border': 1,
        'text_wrap': True
    })

    header_wrap_left_colored = workbook.add_format({
        **font_base,
        'align': 'left',
        'valign': 'vcenter',
        'bg_color': '#D9D9D9',
        'border': 1,
        'text_wrap': True
    })

    highlight_yellow = workbook.add_format({
        **font_base,
        'bg_color': '#FFFF00',
        'valign': 'vcenter',
        'border': 1,
        'align': 'left'
    })

    border = workbook.add_format({
        **font_base,
        'border': 1,
        'valign': 'vcenter',
    })

    border_left = workbook.add_format({
        **font_base,
        'border': 1,
        'valign': 'vcenter',
        'align': 'left'
    })

    border_left_percent = workbook.add_format({
        **font_base,
        'num_format': '0%',
        'border': 1,
        'valign': 'vcenter',
        'align': 'left'
    })

    border_center = workbook.add_format({
        **font_base,
        'border': 1,
        'valign': 'vcenter',
        'align': 'center'
    })

    border_center_percent = workbook.add_format({
        **font_base,
        'num_format': '0.0%',
        'border': 1,
        'valign': 'vcenter',
        'align': 'center',
    })

    normal_right_percent = workbook.add_format({
        **font_base,
        'num_format': '0.0%',
        'valign': 'vcenter',
        'align': 'right',
        'border': 1
        
    })

    bold_right_percent = workbook.add_format({
        'num_format': '0.0%',
        'align': 'right',
        'valign': 'vcenter',
        'bold': True,
        'border': 1,
    })

    normal_right = workbook.add_format({
        **font_base,
        'valign': 'vcenter',
        'align': 'right',
        'border': 1
    })

    bold_right = workbook.add_format({
        **font_base,
        'valign': 'vcenter',
        'align': 'right',
        'border': 1,
        'bold': True  
    })

    # Row 2
    worksheet.write('B2', 'Product Name', highlight_yellow)
    worksheet.write('C2', input_data['product_name'], border_left)
    worksheet.write('D2', 'Unique Customers', header_wrap_left_colored)
    worksheet.write('E2', input_data['unique_customers'], border_left)
    worksheet.write('F2', 'Versions Analysed', header_wrap_left_colored)
    worksheet.write('G2', input_data['versions_analysed'], border_left)
    worksheet.write('H2', 'SaaS Customers Count', header_wrap_left_colored)
    worksheet.write('I2', input_data['saas_customers_count'], border_left)

    worksheet.write('B3', 'Date Range for Cases Analysis', header_wrap_left_colored)
    worksheet.write('C3', input_data['data_range'], border_left)
    worksheet.write('D3', 'Total Cases', header_wrap_left_colored)
    worksheet.write('E3', input_data['total_cases'], border_left)
    worksheet.write('F3', 'Open Cases', header_wrap_left_colored)
    worksheet.write('G3', input_data['open_cases'], border_left)
    worksheet.write('H3', 'Closed Cases', header_wrap_left_colored)
    worksheet.write('I3', input_data['closed_cases'], border_left)

    worksheet.write('B4', 'Total Top 10 LV1 Categories - Case Reduction Count', header_wrap_left_colored)
    worksheet.write('C4', input_data['case_reduction_count_top_10_lv1_categories'], border_left)
    worksheet.write('D4', input_data['case_reduction_percent_top_10_lv1_categories'] / 100.0, border_left_percent)
    worksheet.write('E4', 'LV1', header_wrap_left_colored)
    worksheet.write('F4', 'LV2', header_wrap_left_colored)
    worksheet.write('G4', 'LV3', header_wrap_left_colored)

    worksheet.write('B5', 'Total Top 10 LV3 Categories - Case Reduction Count', header_wrap_left_colored)
    worksheet.write('C5', input_data['case_reduction_count_top_10_categories'], border_left)
    worksheet.write('D5', input_data['case_reduction_percent_top_10_categories'] / 100.0, border_left_percent)
    worksheet.write('E5', input_data['categories_count']['LV1'], border_left)
    worksheet.write('F5', input_data['categories_count']['LV2'], border_left)
    worksheet.write('G5', input_data['categories_count']['LV3'], border_left)

    i=51
    i += 2
    # Categorization
    worksheet.write(f'B{i}', 'Categorization', highlight_yellow)
    worksheet.write(f'C{i}', 'By Top Case Type', border)

    i += 1
    lv1_bar_chart_input_row=i
    worksheet.set_row(i-1, 30) 
    worksheet.write(f'B{i}', 'High Level Top 10 Categories (LV1)', header_wrap_colored)
    worksheet.write(f'C{i}', 'Cases', header_wrap_colored)
    worksheet.write(f'D{i}', 'Contribution on Total Inflow', header_wrap_colored)

    top10_LV1_total_cases=0
    top10_LV1_total_per=0
    for row in input_data['high_level_top_10_categories']:
        i += 1
        worksheet.write(f'B{i}', row['category'], border)
        worksheet.write(f'C{i}', row['cases'], border_center)
        worksheet.write(f'D{i}', row['contribution_on_total_inflow'] / 100.0, border_center_percent)
        top10_LV1_total_cases += row['cases']
        top10_LV1_total_per += row['contribution_on_total_inflow']

    i += 1
    worksheet.write(f'B{i}', 'TOP 10 TOTAL', bold_right)
    worksheet.write(f'C{i}', top10_LV1_total_cases, bold_right)
    worksheet.write(f'D{i}', top10_LV1_total_per/100, bold_right_percent)

    i=90
    bar_chart_input_row = i
    worksheet.set_row(i-1, 30) 
    worksheet.write(f'B{i}', 'Granular Level Top 10 Categories (LV3)', header_wrap_colored)
    worksheet.write(f'C{i}', 'Cases', header_wrap_colored)
    worksheet.write(f'D{i}', 'Average Resolution (in days)', header_wrap_colored)
    worksheet.write(f'E{i}', 'Median Resolution Days', header_wrap_colored)
    worksheet.write(f'F{i}', 'Contribution on Total Inflow', header_wrap_colored)
    worksheet.write(f'G{i}', 'Impact on SaaS Customers Inflow', header_wrap_colored)
    worksheet.write(f'H{i}','Case Severity',header_wrap_colored)

    top10_LV3_total_cases=0
    top10_LV3_total_per=0
    
    for row in input_data['granular_level_top_10_categories']:
        i += 1
        worksheet.write(f'B{i}', row['category'], border)
        worksheet.write(f'C{i}', row['cases'], border_center)
        worksheet.write(f'D{i}', row['average_resolution_in_days'], border_center)
        worksheet.write(f'E{i}', row['median_resolution_in_days'], border_center)
        worksheet.write(f'F{i}', row['contribution_on_total_inflow'] / 100.0, border_center_percent)
        worksheet.write(f'G{i}', row['impact_on_saas_customers_inflow'] / 100.0, border_center_percent)
        worksheet.write(f'H{i}',row['case_severity'],border)
        top10_LV3_total_cases += row['cases']
        top10_LV3_total_per += row['contribution_on_total_inflow']

    i += 1
    worksheet.write(f'B{i}', 'TOP 10 LV3 TOTAL', bold_right)
    worksheet.write(f'C{i}', top10_LV3_total_cases, bold_right)
    worksheet.write(f'F{i}', top10_LV3_total_per/100, bold_right_percent)
    worksheet.write(f'D{i}', '', bold_right)
    worksheet.write(f'E{i}', '', bold_right)
    worksheet.write(f'G{i}', '', bold_right)
    worksheet.write(f'H{i}','',bold_right)

    # Prevention Opportunities
    i += 2
    worksheet.write(f'B{i}', ' Prevention Opportunities', highlight_yellow)
    worksheet.write(f'C{i}', 'By Category', border)

    i += 1
    worksheet.set_row(i-1, 30) 
    worksheet.write(f'B{i}', 'Contributors', header_wrap_colored)
    worksheet.write(f'C{i}', 'Total Cases', header_wrap_colored)
    worksheet.write(f'D{i}', 'Potential Impact on Inflow%', header_wrap_colored)
    worksheet.write(f'E{i}', 'Top 10 Total Cases', header_wrap_colored)
    worksheet.write(f'F{i}', 'Potential Impact on Inflow%', header_wrap_colored)

    for row in input_data['contributors']:
        i += 1
        worksheet.write(f'B{i}', row['contributor'], border)
        worksheet.write(f'C{i}', row['total_cases'], border_center)
        worksheet.write(f'D{i}', row['potential_impact'] / 100.0, border_center_percent)
        worksheet.write(f'E{i}', row['top_20_cases'], border_center)
        worksheet.write(f'F{i}', row['top_20_impact']/100.0, border_center_percent)

    # contributor
    estimated_lines=0
    contributor_top20_list = input_data['contributor_top20_list']
    for contributor_top20 in contributor_top20_list:
        for key, value in contributor_top20.items():
            
            i += 2
            worksheet.write(f'B{i}', key, highlight_yellow)
            worksheet.write(f'C{i}', 'Top 10', border)

            i += 1
            worksheet.merge_range(f'B{i}:G{i}', key, header_colored)
            worksheet.write(f'H{i}', 'Case Count', header_colored)
            worksheet.write(f'I{i}', 'Business Impact', header_colored)
            worksheet.write(f'J{i}','Case Severity',header_colored)

            cont_sum = 0
            cont_case_sum = 0
            for row in value:
                i += 1
                estimated_lines = estimate_lines(row['title'], 155)
                row_height = estimated_lines * 15
                worksheet.set_row(i-1, row_height)
                worksheet.merge_range(f'B{i}:G{i}', row['title'], wrap_left)
                worksheet.write(f'H{i}', row['case_count'], normal_right)
                worksheet.write(f'I{i}', row['business_impact'] / 100.0, normal_right_percent)
                worksheet.write(f'J{i}',row['case_severity'],border)
                cont_sum += row['business_impact']
                cont_case_sum += row['case_count']
    
            i += 1
            worksheet.merge_range(f'B{i}:G{i}', 'TOTAL IMPACT', bold_right)
            worksheet.write(f'H{i}', cont_case_sum, bold_right)
            worksheet.write(f'I{i}', cont_sum/100, bold_right_percent)
            worksheet.write(f'J{i}', '', bold_right)

    # Accounts Across Categories
    i += 2
    worksheet.write(f'B{i}', 'Accounts Across Categories', highlight_yellow) 
    worksheet.write(f'C{i}', 'Top 20', border)

    i += 1
    worksheet.merge_range(f'B{i}:D{i}', 'Account Name', header_colored)
    worksheet.merge_range(f'E{i}:H{i}', 'Category Names', header_colored)
    worksheet.write(f'I{i}', 'Num_LV3_Categories', header_colored)
    worksheet.write(f'J{i}', 'Category Coverage %', header_colored)

    automation_sum = 0
    automation_case_sum = 0
    for row in input_data['account_categories']:
        i += 1
        row_height = estimated_lines * 15
        worksheet.set_row(i-1, row_height)
        worksheet.merge_range(f'B{i}:D{i}', row['AccountName'], wrap_left)
        worksheet.merge_range(f'E{i}:H{i}', row['LV3Categories'], wrap_left)
        worksheet.write(f'I{i}', row['Count'], normal_right)
        worksheet.write(f'J{i}', row['CategoryCoverage'] / 100.0, normal_right_percent)
    
    # Top Case contributors
    i += 2
    worksheet.write(f'B{i}', 'Top Case contributors', highlight_yellow)
    worksheet.write(f'C{i}', 'Top 20', border)

    i += 1
    worksheet.merge_range(f'B{i}:G{i}', 'Account Name', header_colored)
    worksheet.write(f'H{i}', 'Case Count', header_colored)
    worksheet.write(f'I{i}', 'Percentage', header_colored)

    enhancement_sum = 0
    enhancement_case_sum = 0
    for row in input_data['account_case']:
        i += 1
        worksheet.set_row(i-1, row_height)
        worksheet.merge_range(f'B{i}:G{i}', row['AccountName'], wrap_left)
        worksheet.write(f'H{i}', row['case_count'], normal_right)
        worksheet.write(f'I{i}', row['Percentage'] / 100.0, normal_right_percent)
    
    i += 1
    
    # LV1 GRAPH
    lv1_bar_start_row = lv1_bar_chart_input_row
    lv1_bar_end_row = lv1_bar_start_row + len(input_data['high_level_top_10_categories']) - 1

    bar_chart = workbook.add_chart({'type': 'column'}) 

    bar_chart.add_series({
        'name': 'Case inflow by categories',
        'categories': ['Summary', lv1_bar_start_row, 1, lv1_bar_end_row, 1],
        'values': ['Summary', lv1_bar_start_row, 3, lv1_bar_end_row, 3],
        'data_labels': {'value': True},
    })

    bar_chart.set_title({'name': 'Case inflow by LV1 categories'})
    bar_chart.set_x_axis({'name': 'Categories'})
    bar_chart.set_y_axis({'name': 'Impact'})

    # Insert chart at desired location
    bar_chart.set_size({
        'width': 1500,   # in pixels
        'height': 450   # in pixels
    })

    worksheet.insert_chart('B28', bar_chart)

    i += 1
    bar_start_row = bar_chart_input_row
    bar_end_row = bar_start_row + len(input_data['granular_level_top_10_categories']) - 1

    bar_chart = workbook.add_chart({'type': 'column'}) 

    bar_chart.add_series({
        'name': 'Case inflow by categories',
        'categories': ['Summary', bar_start_row, 1, bar_end_row, 1],
        'values': ['Summary', bar_start_row, 5, bar_end_row, 5],
        'data_labels': {'value': True},
    })

    bar_chart.set_title({'name': 'Case inflow by LV3 categories'})
    bar_chart.set_x_axis({'name': 'Categories'})
    bar_chart.set_y_axis({'name': 'Impact'})

    # Insert chart at desired location
    bar_chart.set_size({
        'width': 1500,   # in pixels
        'height': 450   # in pixels
    })

    worksheet.insert_chart('B66', bar_chart)
    
    # Account wise LV3
    i=1
    worksheet4 = workbook.add_worksheet('Account Categories')
    worksheet4.set_column('A:A', 50, wrap)
    worksheet4.set_column('B:B', 100, wrap)
    worksheet4.set_column('C:C', 20, wrap)
    worksheet4.write(f'A{i}', 'Account Name', header_colored)
    worksheet4.write(f'B{i}', 'LV3 Categories', header_colored)
    worksheet4.write(f'C{i}', 'Count', header_colored)

    for row in input_data['account_categories_full']:
        i += 1
        worksheet4.write(f'A{i}', row['AccountName'], wrap_left)
        worksheet4.write(f'B{i}', row['LV3Categories'], wrap_left)
        worksheet4.write(f'C{i}', row['Count'], normal_right)
    
    # Year/Month Case count
    worksheet3 = workbook.add_worksheet("Case Trend")

    # Write headers
    months_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    worksheet3.write(0, 0, "Month")
    for i, month in enumerate(months_order):
        worksheet3.write(i + 1, 0, month)

    # Write year-wise values in columns
    for col, entry in enumerate(input_data['year_month_case_count'], start=1):
        worksheet3.write(0, col, str(entry["year"]))
        for row, month in enumerate(months_order, start=1):
            value = entry["months"].get(month)
            worksheet3.write(row, col, value if value is not None else "")

    # Line Chart
    chart = workbook.add_chart({'type': 'line'})

    # Add a series for each year
    for col, entry in enumerate(input_data['year_month_case_count'], start=1):
        chart.add_series({
            'name':       [worksheet3.name, 0, col],
            'categories': [worksheet3.name, 1, 0, 12, 0],
            'values':     [worksheet3.name, 1, col, 12, col],
            'line':       {'width': 1.5}
        })

    # Configure the chart axes and title
    chart.set_title({'name': 'Case Inflow- Monthly Trend'})
    chart.set_x_axis({'name': 'Month'})
    chart.set_y_axis({'name': 'Count'})
    chart.set_legend({'position': 'bottom'})

    chart.set_size({
        'width': 600,   # in pixels
        'height': 400   # in pixels
    })

    # Insert the chart into the worksheet
    worksheet.insert_chart('B7', chart)
    workbook.close()

def clean_and_eval(x):
    if isinstance(x, str):
        try:
            # Optional cleanup: remove leading commas or bad formatting
            x = re.sub(r'^\[\s*,\s*', '[', x)  # fix "[,1,2,3" => "[1,2,3"
            return ast.literal_eval(x)
        except (SyntaxError, ValueError):
            return []  # or None if preferred
    return x

def reprot_process(product_name, cases_df, output_file_path):
    
    input_data = dict()
    
    input_data['product_name'] = cases_df['Product Line'].dropna().unique()[0]
    input_data['unique_customers'] = cases_df['Account Name'].nunique()
    input_data['versions_analysed'] = ','.join(cases_df['Product Version Name'].dropna().unique().astype(str))
    input_data['saas_customers_count'] = cases_df[cases_df['Has SaaS Asset flag'] == True]['Account Name'].nunique()
    
    # Convert date columns to datetime objects
    cases_df['Date/Time Opened'] = pd.to_datetime(cases_df['Date/Time Opened'], errors='coerce')
    cases_df['Date/Time Closed'] = pd.to_datetime(cases_df['Date/Time Closed'], errors='coerce')
    input_data['data_range'] = f"{cases_df['Date/Time Opened'].min().date()} to {cases_df['Date/Time Opened'].max().date()}"
    input_data['total_cases'] = len(cases_df)
    input_data['open_cases'] = len(cases_df[cases_df['Status'].str.lower() != 'closed'])
    input_data['closed_cases'] = len(cases_df[cases_df['Status'].str.lower() == 'closed'])

    # Load the Excel file into analysis_df
    analysis_df = pd.read_excel(output_file_path, sheet_name='Detailed Output')
    analysis_df.columns = analysis_df.columns.str.strip()

    distinct_counts = {
        "LV1": analysis_df["LV1"].nunique(),
        "LV2": analysis_df["LV2"].nunique(),
        "LV3": analysis_df["LV3"].nunique()
    }
    input_data['categories_count']=distinct_counts
    
    # Group by LV1 and sum the case_count
    lv1_summary = analysis_df.groupby('LV1')['Case Count'].sum().reset_index()

    # Sort by case_count in descending order
    lv1_top10 = lv1_summary.sort_values(by='Case Count', ascending=False).head(10).copy()


    # Compute total number of cases for inflow calculation
    total_cases = len(cases_df)

    # Step 1: Normalize Case Numbers
    analysis_df.columns = analysis_df.columns.str.strip()
    cases_df['Case Number'] = cases_df['Case Number'].astype(str)

    # Step 2: Create set of SaaS case numbers
    saas_case_set = set(
        cases_df[cases_df['Has SaaS Asset flag'] == True]['Case Number'].dropna()
    )

    # Step 3: Convert case_numbers column from string to list
    analysis_df['Case Numbers'] = analysis_df['Case Numbers'].apply(clean_and_eval)

    # Step 4: Compute SaaS % impact for each LV3 category based on number of SaaS cases
    def saas_impact_percentage(case_list):
        if  not case_list: # or pd.isna(case_list):
            return 0.0
        total = len(case_list)
        saas_count = sum(1 for case in case_list if str(case) in saas_case_set)
        return round((saas_count / total) * 100, 2)

    # Step 5: Apply to analysis_df
    analysis_df['Impact on SaaS Customers Inflow'] = analysis_df['Case Numbers'].apply(saas_impact_percentage)

    # Step 6: Sort and extract top 10 LV3 categories by number of cases
    top10_df = analysis_df.sort_values(by='Case Count', ascending=False).head(10).copy()
    top10_total_cases = top10_df['Case Count'].sum()

    # Step 7: Add summary to input_data
    input_data['case_reduction_count_top_10_categories'] = top10_total_cases
    input_data['case_reduction_percent_top_10_categories'] = (top10_total_cases / input_data['closed_cases']) * 100

    # Step 8: Build JSON for top 10 categories
    top10_json = []
    for _, row in top10_df.iterrows():
        if pd.isna(row['Case Count']): continue
        
        top10_json.append({
            "category": row['LV3'],
            "cases": int(row['Case Count']),
            "average_resolution_in_days": float(row['Average Resolution Days']),
            "median_resolution_in_days": float(row['Median Resolution Days']),
            "percentile_95_resolution_in_days": float(row['95th Percentile Resolution Days']),
            "contribution_on_total_inflow": round(row['Case Count'] / input_data['closed_cases'] * 100, 2),
            "impact_on_saas_customers_inflow": row['Impact on SaaS Customers Inflow'],
            "case_severity":row['Case Severity Distribution']
        })

    input_data['granular_level_top_10_categories'] = top10_json

    top10_lv1_total_cases = lv1_top10['Case Count'].sum()
    input_data['case_reduction_count_top_10_lv1_categories'] = top10_lv1_total_cases
    input_data['case_reduction_percent_top_10_lv1_categories'] = top10_lv1_total_cases/input_data['closed_cases']*100

    # Generate the JSON output
    top10_LV1_json = []
    for _, row in lv1_top10.iterrows():
        top10_LV1_json.append({
            "category": row['LV1'],
            "cases": int(row['Case Count']),
            "contribution_on_total_inflow": round(row['Case Count'] / input_data['closed_cases'] * 100, 2),
            "impact_on_saas_customers_inflow": int(row.get('Impact on SaaS Customers Inflow', 0))
        })

    input_data['high_level_top_10_categories'] = top10_LV1_json
    
    # Mapping of analysis_df columns to contributor labels
    contributor_category = analysis_df['Insight Category'].unique().tolist()
    contributor_category = [c for c in contributor_category if pd.notna(c) and str(c).strip() and c != "Other"]

    # Build contributors list dynamically
    top_count=10
    contributors = []
    for category in contributor_category:
        filtered = analysis_df[analysis_df['Insight Category'] == category]
        total = filtered['Case Count'].sum()

        top_20_filtered = filtered.sort_values(by='Case Count', ascending=False).head(top_count)
        # Sum of top 20 cases
        top_20_cases = top_20_filtered['Case Count'].sum()
        contributors.append({
            'contributor': category,
            'total_cases': int(total),
            'potential_impact': int(total)/total_cases*100,  # Can be calculated if needed
            'top_20_cases':int(top_20_cases),
            'top_20_impact':round((top_20_cases / input_data['closed_cases']) * 100 if total_cases > 0 else 0,2)
        })

    # Add to input_data
    input_data['contributors'] = contributors

    contributor_top20_list=[]
    for category in contributor_category:
        cont_dict =  dict()
        cont_df = pd.read_excel(output_file_path, sheet_name=category)

        cont_df.columns = cont_df.columns.str.strip()
        cont_df_sorted = cont_df.sort_values(by='Case Count', ascending=False)
        cont_top_20 = cont_df_sorted.head(top_count)

        cont_top20_json = []
        for _, row in cont_top_20.iterrows():
            title = 'KB Title' if category == 'Knowledge Base Candidate' else category
            cont_top20_json.append({
                "title": row[title],
                "case_count": int(row['Case Count']),
                "business_impact": round(row['Case Count'] / input_data['closed_cases'] * 100, 2),
                "case_severity":row['Case Severity Distribution']
            })
        cont_dict[category]=cont_top20_json
        contributor_top20_list.append(cont_dict)

    input_data['contributor_top20_list'] = contributor_top20_list

    # Create a Case Numbers → Account Name mapping
    # case_to_account = cases_df.set_index('Case Number')['Account Name'].to_dict()
    filtered_df = cases_df[
        cases_df['Account Name'].notna() & 
        (cases_df['Account Name'].astype(str).str.strip() != '') & 
        cases_df['Case Number'].notna()
    ]

    case_to_account = filtered_df.set_index('Case Number')['Account Name'].to_dict()
    
    # Map LV3 categories to each account
    account_lv3_map = {}

    for _, row in analysis_df.iterrows():
        lv3_category = row['LV3']
        case_numbers = row['Case Numbers']

        if pd.isna(lv3_category):
            continue  # Skip invalid LV3 values

        for case in case_numbers:
            account = case_to_account.get(str(case))
            if account:
                account_lv3_map.setdefault(account, set()).add(lv3_category)

    # Build output structure
    output_data = []
    for account, lv3_set in account_lv3_map.items():
        output_data.append({
            'Account Name': account,
            'LV3 Categories': ', '.join(sorted(lv3_set)),
            'Count': len(lv3_set)
        })

    # Create DataFrame and sort
    result_df = None
    if output_data:
        result_df = pd.DataFrame(output_data)
    else:
        result_df = pd.DataFrame(columns=["Account Name", "LV3 Categories", "Count"])
    result_df = result_df.sort_values(by='Count', ascending=False).reset_index(drop=True)

    # Append total unique LV3 categories at the end
    total_unique_lv3 = analysis_df['LV3'].nunique()
    total_row = pd.DataFrame([{
        'Account Name': 'TOTAL',
        'LV3 Categories': '',
        'Count': total_unique_lv3
    }])

    acc_cat_df_top_20 = result_df.head(20)
    acc_cat_df_top20_json = []
    for _, row in acc_cat_df_top_20.iterrows():
        acc_cat_df_top20_json.append({
            "AccountName": row['Account Name'],
            "LV3Categories": row['LV3 Categories'],
            "Count": round(row['Count'] ),
            "CategoryCoverage":round(row['Count'] / total_unique_lv3 * 100, 2)
        })
    
    input_data['account_categories']=acc_cat_df_top20_json

    acc_cat_json = []
    for _, row in result_df.iterrows():
        acc_cat_json.append({
            "AccountName": row['Account Name'],
            "LV3Categories": row['LV3 Categories'],
            "Count": round(row['Count'] )
        })
    
    input_data['account_categories_full']=acc_cat_json

    # Step 2: Clean column names
    cases_df.columns = cases_df.columns.str.strip()
    # Step 3: Group by 'Account Name' and count
    account_case_distribution_df = (
        cases_df[cases_df['Account Name'].notna() & (cases_df['Account Name'].str.strip() != '')]
        .groupby('Account Name')
        .size()
        .reset_index(name='Case Count')
        .sort_values(by='Case Count', ascending=False)
    )

    # Step 4: Convert to JSON format (list of dicts)
    account_case_distribution_json = account_case_distribution_df.head(20)
    acc_case_df_top20_json = []
    for _, row in account_case_distribution_json.iterrows():
        acc_case_df_top20_json.append({
            "AccountName": row['Account Name'],
            "case_count": row['Case Count'],
            "Percentage": round(row['Case Count'] / input_data['closed_cases'] * 100, 2)
        })
    
    input_data['account_case']=acc_case_df_top20_json

    # Step 2: Convert to datetime
    cases_df['Date/Time Opened'] = pd.to_datetime(cases_df['Date/Time Opened'], errors='coerce')

    # Step 3: Extract Year and Month
    cases_df['Year'] = cases_df['Date/Time Opened'].dt.year
    cases_df['Month'] = cases_df['Date/Time Opened'].dt.strftime('%b')  # e.g., Jan, Feb

    # Step 4: Define proper month order
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    cases_df['Month'] = pd.Categorical(cases_df['Month'], categories=month_order, ordered=True)

    # Step 5: Group by Year and Month
    year_month_summary = (
        cases_df.groupby(['Year', 'Month'],observed=True)
        .size()
        .reset_index(name='Total Cases')
        .sort_values(by=['Year', 'Month'])
        .reset_index(drop=True)
    )

    # Step 6: Build structured dictionary
    input_data['year_month_case_count'] = []

    for year in year_month_summary['Year'].unique():
        # Filter for the current year
        months_data = year_month_summary[year_month_summary['Year'] == year]
        
        # Create a dict for the month's case counts
        month_dict = dict(zip(months_data['Month'], months_data['Total Cases']))
        
        # Ensure all months exist (fill with None if missing)
        full_month_dict = {month: month_dict.get(month, None) for month in month_order}
        
        input_data['year_month_case_count'].append({
            "year": int(year),
            "months": full_month_dict
        })
    
    #Escalation trend
    # Step 1: Filter cases with status = 'Escalated' (case-insensitive + clean spaces)
    escalated_df = cases_df[cases_df['Status'].str.strip().str.lower() == 'escalated'].copy()

    # Step 2: Convert date if not already
    escalated_df['Date/Time Opened'] = pd.to_datetime(escalated_df['Date/Time Opened'], errors='coerce')

    # Step 3: Extract Year and Month
    escalated_df['Year'] = escalated_df['Date/Time Opened'].dt.year
    escalated_df['Month'] = escalated_df['Date/Time Opened'].dt.strftime('%b')

    # Step 4: Set categorical month order
    escalated_df['Month'] = pd.Categorical(escalated_df['Month'], categories=month_order, ordered=True)

    # Step 5: Group by Year and Month
    escalated_trend = (
        escalated_df.groupby(['Year', 'Month'], observed=True)
        .size()
        .reset_index(name='Escalated Cases')
        .sort_values(by=['Year', 'Month'])
        .reset_index(drop=True)
    )
    input_data['year_month_escalated_case_count'] = []

    for year in escalated_trend['Year'].unique():
        # Filter for the current year
        months_data = escalated_trend[escalated_trend['Year'] == year]
        
        # Create a dict for escalated case counts
        month_dict = dict(zip(months_data['Month'], months_data['Escalated Cases']))
        
        # Fill missing months with None
        full_month_dict = {month: month_dict.get(month, None) for month in month_order}
        
        input_data['year_month_escalated_case_count'].append({
            "year": int(year),
            "months": full_month_dict
        })

    summary_report(product_name, input_data)
