
# Ensure to download these first:
        # !pip install spacy==3.7.2
        # pip install spacy-transformers
        # !python -m spacy download en_core_web_trf
        # pip install pandas
        


import spacy
import re
import pandas as pd
import email.utils

nlp = spacy.load("en_core_web_trf")


def find_name(df):
    name = []
    for i in range(171):
        t = df.at[i, 'parsedTxt']
        if type(t) != float:
            doc = nlp(t.strip().replace('\n', ' '))
            name.append([(ent.text) for ent in doc.ents if ent.label_ in ["PERSON"]])
        else:
            name.append([])

    l_name = []
    for i in range(171):
        if not name[i]:
            l_name.append("")
        else:
            l_name.append(name[i][0])

    dict1 = {'fullname': l_name}
    return dict1

def find_email_addresses(text):
    email_pattern = r'[\w\.-]+@[\w\.-]+'
    potential_addresses = re.findall(email_pattern, text)

    valid_email_addresses = []
    for address in potential_addresses:
        try:
            email.utils.parseaddr(address)
            valid_email_addresses.append(address)
        except:
            pass

    while len(valid_email_addresses) < 2:
        valid_email_addresses.append("")

    return valid_email_addresses

def find_phone_numbers(text):
    phone_pattern = r'(\+91[- ]?\d{5}[- ]?\d{5}|[0-9]{10}|[0-9]{4}[- ][0-9]{10}|\+91[- ]?[0-9]{5}[- ]?[0-9]{5}|\+[0-9]{1,5}[- ]?[0-9]{10})'
    phone_numbers = re.findall(phone_pattern, text)

    valid_phone_numbers = []
    for number in phone_numbers:
        if re.match(phone_pattern, number):
            valid_phone_numbers.append(number)

    while len(valid_phone_numbers) < 2:
        valid_phone_numbers.append("")

    return valid_phone_numbers

def find_website(text):
    urls = re.findall(r'(http[s]?://)?(www\.)?([a-zA-Z0-9.-]+)\.([a-zA-Z]{2,4})|www\.[a-zA-Z0-9.-]+\s', text)
    website = ""
    for url in urls:
        website = url[0] + url[1] + url[2] + '.' + url[3]

    return website

def find_company(df):
    l_company = []
    for i in range(171):
        t = df.at[i, 'parsedTxt']
        potential_company = ""
        if type(t) != float:
            doc = nlp(t)
            company_entities = [ent.text for ent in doc.ents if ent.label_ in ["ORG"]]
            potential_company = " ".join(company_entities)
            l_company.append(potential_company)
        else:
            l_company.append("")

    dict6 = {'company': l_company}
    return dict6

def find_address(df):
    l_address = []
    for i in range(171):
        t = df.at[i, 'parsedTxt']
        potential_address = ""
        if type(t) != float:
            doc = nlp(t)
            address_entities = [ent.text for ent in doc.ents if ent.label_ in ["GPE", "FAC", "LOC"]]
            potential_address = " ".join(address_entities)
            l_address.append(potential_address)
        else:
            l_address.append("")

    dict5 = {'address': l_address}
    return dict5

def making_dataframes(df):
    df1 = pd.DataFrame(find_name(df))

    l_email = []
    for i in range(171):
        t = df.at[i, 'parsedTxt']
        if type(t) != float:
            l_email.append(find_email_addresses(t))
        else:
            l_email.append(["", ""])

    dict2 = {'email': [l_email[j][0] for j in range(171)],
             'email_2': [l_email[j][1] for j in range(171)]}
    df2 = pd.DataFrame(dict2)

    l_phone = []
    for i in range(171):
        t = df.at[i, 'parsedTxt']
        if type(t) != float:
            l_phone.append(find_phone_numbers(t))
        else:
            l_phone.append(["", ""])

    dict3 = {'phone': [l_phone[j][0] for j in range(171)],
             'phone_2': [l_phone[j][1] for j in range(171)]}
    df3 = pd.DataFrame(dict3)

    l_website = []
    for i in range(171):
        t = df.at[i, 'parsedTxt']
        if type(t) != float:
            l_website.append(find_website(t))
        else:
            l_website.append(find_website(""))

    dict4 = {'website': l_website}
    df4 = pd.DataFrame(dict4)

    df5 = pd.DataFrame(find_company(df))
    df6 = pd.DataFrame(find_address(df))

    merged_dataframe = pd.concat([df1, df2, df3, df4, df5, df6], axis = 1)
    return merged_dataframe

def main():
    df = pd.read_excel('MyContacts.xlsx')
    display(making_dataframes(df))

main()
