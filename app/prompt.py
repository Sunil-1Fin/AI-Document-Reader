
import json

def get_prompt():
    json_template = json.dumps(template_dict(), indent=2)
    prompt = f"""
        Based on the given Context (text from Form 16 PDF), respond by filling in the provided JSON template.

        Instructions:
        - Use the template strictly as-is. Do NOT add or remove any keys.
        - Do NOT change the structure. Only fill the 'answer' fields.
        - If a value is not found in the context, leave the answer as-is.
        - Never hallucinate or guess.
        - Some questions include a 'note' to help locate values.
        - Answer ONLY using values present in the context.

        Context:
        {context}

        Template:
        {json_template}
        """


def template_dict():
    
    template_dict = {
        "form_16":{
            "question":"Is this PDF Form 16?",
            "answer":"Yes or No"
        },
        "form_16_part_a": {
            "question": "Is this Form has Part A data",
            "answer": "yes or no"
        },
        "form_16_part_b": {
            "question": "Is this Form has Part B data",
            "answer": "yes or no"
        },
        "financial_year_from":{
            "question":"Financial year from?",
            "answer":"02-Apr-1988"
        },
        "financial_year_to":{
            "question":"Financial year to?",
            "answer":"02-Apr-1988"
        },
        "assessment_year":{
            "question":"Assessment Year",
            "answer":"2023-24"
        },
        "period_with_employer_from":{
            "question":" Period with the Employer From",
            "answer":"1-Apr-2023"
        },
        "period_with_employer_to":{
            "question":" Period with the Employer To",
            "answer":"1-Apr-2023"
        },
        "employer_state":{
            "question":"Employer State",
            "answer":""
        },
        "employer_city":{
            "question":"Employer City",
            "answer":"0"
        },
        
        "employer_name":{
            "question":"Name of the Employer",
            "answer":""
        },
        "employer_tan":{
            "question":"TAN Number of Employer",
            "answer":""
            },
        "employer_pan":{
            "question":"PAN Number of Employer",
            "answer":""
            },
        "employee_pan":{
            "question":"PAN Number of Employee",
            "answer":""
            },
        "address_of_employer":{
            "question":"Address of the Employer",
            "answer":""
        },
        "email_of_employer":{
            "question":"Email of Employer",
            "answer":""
        },
        "pincode_of_employer":{
            "question":"Pincode of Employer",
            "answer":""
        },
            "details_of_tax_deducted_and_deposited":{
            "question":" DETAILS OF TAX DEDUCTED AND DEPOSITED IN THE CENTRAL GOVERNMENT ACCOUNT THROUGH CHALLAN, required format in answer.",
            "answer":[{"sr_no":1,"tax_deposited":"0","bsr_code":"0","date_on_tax_deposited":"date","chalan_serial_number":"","status_of_matching_with_oltlas":""}]
        },
        "details_of_tax_deducted_and_deposited_total":{
            "question":" total of DETAILS OF TAX DEDUCTED AND DEPOSITED IN THE CENTRAL GOVERNMENT ACCOUNT THROUGH CHALLAN",
            "answer": "0"
        },
        "summary_of_amount_paid":{
            "question":"Summary of amount paid/credited and tax deducted at source thereon in respect of the employee,required format in answer.",
            "answer": [{"quarter":"","recipt_number":"","amount_paid_cedited":"0","amount_tax_deducted":"0","amount_tax_deposited_remitted":"0"}]
        },
        "summary_of_amount_paid_total":{
            "question":"Total of Summary of amount paid/credited and tax deducted at source thereon in respect of the employee,required format in answer.",
            "answer":"0"
        },
        "salary_as_provision_section_17_1":{
            "question":"A. 1. (a). Salary as per provisions contained in section 17 (1)",
            "answer":"0"
        },
        "salary_as_provision_section_17_2":{
            "question":" A. 1. (b). Value of perquisites under section 17(2) (as per Form No. 12BA, wherever applicable)",
            "answer":"0"
        },
        "salary_as_provision_section_17_3":{
            "question":"A. 1. (c). Profits in lieu of salary under section 17(3) (as per Form No.12BA, wherever applicable)",
            "answer":"0"
        },
         "total_gross_salary":{
            "question":"A. 1. (d). Total of Gross Salary",
            "answer":"0"
        },
         "reported_total_amount_salary_received":{
            "question":"A. 1. (e). Reported total amount of salary received from other employer(s)",
            "answer":"0"
        },
         "travel_concession_section_10_5":{
            "question":"A. 2. (a). Travel concession or assistance under section 10(5)",
            "answer":"0"
        },
        "death_retirement_gratuity_section_10_10":{
            "question":"A. 2. (b). Death-cum-retirement gratuity under section 10(10)",
            "answer":"0"
        },
        "commuted_value_section_10_10a":{
            "question":"A. 2. (c). Commuted value of pension under section 10(10A)",
            "answer":"0"
        },
        "cash_equivalent_section_10_10aa":{
            "question":"A. 2. (d). Cash equivalent of leave salary encashment under section 10 (10AA)",
            "answer":"0"
        },
        "house_rent_section_10_13a":{
            "question":"A. 2. (e). House rent allowance under section 10(13A)",
            "answer":"0"
        },
        "other_special_allowence_section_10_14":{
            "question":"A. 2. (f). Other special allowances under section 10(14)",
            "answer":"0"
        },
        "amount_of_any_other_section_10":{
            "question":"A. 2. (g). Amount of any other exemption under section 10 [Note: Break-up to be filled and signed  by employer in the table provide at the bottom of this form]",
            "answer":"0"
        },
        "total_amount_any_other_exemption_section_10":{
            "question":"A. 2. (h). Total amount of any other exemption under section 10",
            "answer":"0"
        },
        "total_amount_exemption_section_10":{
            "question":"A. 2. (i). Total amount of exemption claimed under section 10 [2(a)+2(b)+2(c)+2(d)+2(e)+2(f)+2(h)]",
            "answer":"0"
        },
        "total_amount_recived_from_currunt_employer":{
            "question":"A. 3. Total amount of salary received from current employer[1(d)-2(i)]",
            "answer":"0"
        },
         "standard_deduction_section_16_ia":{
            "question":"A. 4.(a) Standard deduction under section 16(ia)",
            "answer":"0"
        },
        "entertaiment_allownece_16ii":{
            "question":"A. 4.(b) Entertainment allowance under section 16(ii)",
            "answer":"0"
        },
        "tax_on_employment_section_16iii":{
            "question":"A. 4.(c) Tax on employment under section 16(iii)",
            "answer":"0"
        },
        "total_amount_of_deduction":{
            "question":"A. 5 Total amount of deductions under section 16 [4(a)+4(b)+4(c)]",
            "answer":"date"
        },
        "income_chargeable_under_salaries":{
            "question":"A. 6. Income chargeable under the head 'Salaries' [(3+1(e)-5]",
            "answer":"0"
        },
            "income_from_house_property_reported_by_employee":{
            "question":"Income (or admissible loss) from house property reported by employee offered for TDS",
            "note":"value can be after provided question after (a)",
            "answer":"0"
        },
        "income_under_the_other_source_tds":{
            "question":"Income under the head Other Sources offered for TDS",
            "answer":"0"
        },
        "total_amount_of_other_income_reported_by_employee":{
            "question":"A. 8. Total amount of other income reported by the employee [7(a)+7(b)] ",
            "answer":"0"
        },
        "gross_total_income_6_8":{
            "question":"A. 9. Gross total income (6+8)",
            "answer":"0"
        },
        "deduction_of_li_pf_under_80c":{
            "question":"A. 10. (a). Deduction in respect of life insurance premia, contributions to provident fund etc. under section 80C",
            "answer":[{"gross_amount":"0","deductable_amount":"0"}]
        },
        "deduction_under_section_80ccc":{
            "question":"A. 10. (b). Deduction in respect of contribution to certain pension funds under section 80CCC",
            "answer":[{"gross_amount":"0","deductable_amount":"0"}]
        },
        "deduction_under_section_80ccd_1":{
            "question":"A. 10. (c). Deduction in respect of contribution by taxpayer to pension scheme under section 80CCD (1)",
            "answer":[{"gross_amount":"0","deductable_amount":"0"}]
        },
        "total_deduction_under_80c_90ccc_80ccd1":{
            "question":"A. 10. (d). Total deduction under section 80C, 80CCC and 80CCD(1)",
            "answer":[{"gross_amount":"0","deductable_amount":"0"}]
        },
        "deduction_under_section_80ccd_1b":{
            "question":"(e). Deductions in respect of amount paid/deposited to notified pension scheme under section 80CCD (1B)",
            "answer":[{"gross_amount":"0","deductable_amount":"0"}]
        },
        "deduction_under_section_80ccd_2":{
            "question":"(f). Deduction in respect of contribution by Employer to pension scheme under section 80CCD (2)",
            "answer":[{"gross_amount":"0","deductable_amount":"0"}]
        },
        "deduction_under_section_80d":{
            "question":"(g). Deduction in respect of health insurance premia under section 80D",
            "answer":[{"gross_amount":"0","deductable_amount":"0"}]
        },
        "deduction_under_ection_80e":{
            "question":"A. 10. (h). Deduction in respect of interest on loan taken for higher education under section 80E",
            "answer":[{"gross_amount":"0","deductable_amount":"0"}]
        },
        "deduction_under_section_80cch_employer_to_agnipath":{
            "question":"A. 10. (i). Deduction in respect of contribution by the employee to Agnipath Scheme under section 80CCH",
            "answer":[{"gross_amount":"0","deductable_amount":"0"}]
        },
        "dedution_under_80cch_by_central_government":{
            "question":"A. 10. (j). Deduction in respect of contribution by the Central Government to Agnipath Scheme under section 80CCH",
            "answer":[{"gross_amount":"0","deductable_amount":"0"}]
        },
        "deduction_under_section_80g":{
            "question":"A. 10. (k). Total Deduction in respect of donations to certain funds, charitable institutions, etc. under section 80G",
            "answer":[{"gross_amount":"0","qualifing_amount":"0","deductable_amount":"0"}]
        },
        "deduction_under_section80tta":{
            "question":"A. 10. (l). Deduction in respect of interest on deposits in savings account under section 80TTA",
            "answer":[{"gross_amount":"0","qualifing_amount":"0","deductable_amount":"0"}]
        },
        "total_amount_deductable_under_chapter_vi_a":{
            "question":"A. 10. (n). Total of amount deductible under any other provision(s) of Chapter VI-A",
            "answer":[{"gross_amount":"0","qualifing_amount":"0","deductable_amount":"0"}]
        },
        "aggregate_deductable_amount_under_chappter_via":{
            "question":"A. 11. Aggregate of deductible amount under Chapter VI-A [10(d)+10(e)+10(f)+10(g)+10(h)+10(i)+10(j)+10(l)]",
            "answer":"0"
        },
        "total_taxable_income_9_11":{
            "question":"A. 12. Total taxable income (9-11)",
            "answer":"0"
        },
        "tax_on_total_income":{
            "question":"A. 13. Tax on total income",
            "answer":"0"
        },

        "rebate_under_section_87a":{
            "question":"A. 14. Rebate under section 87A, if applicable",
            "answer":"0"
        },
        "surharge_wherever_applicable":{
            "question":"A. 15. Surcharge, wherever applicable",
            "answer":"0"
        },
        "health_and_education_cess":{
            "question":"A. 16. Health and education cess",
            "answer":"0"
        },
        "tax_payable_13_15_16":{
            "question":"A. 17. Tax payable (13+15+16-14)",
            "answer":"0"
        },

        "less_relief_under_section_89":{
            "question":"A. 18. Less: Relief under section 89 (attach details)",
            "answer":"0"
        },
        "net_tax_payable_17_18":{
            "question":"A. 19. Net tax payable (17-18)",
            "answer":"0"
        }
        }
    return template_dict