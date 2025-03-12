#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 3 2025

@author: deni-kun
"""

# Library imports

#import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from functools import reduce
import pymupdf as fitz
import numpy as np
import streamlit as st
import os
import plotly.graph_objects as go
#from PyPDF2 import PdfReader, PdfWriter
#from io import BytesIO
#from reportlab.pdfgen import canvas
import requests
#import shutil
#import subprocess
#from pathlib import Path
from collections import defaultdict
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode
#from PIL import Image

###############################
########## FUNCTIONS ##########
###############################
 
st.set_page_config(layout='centered', page_icon=':octopus:')

def aggrid_interactive_table(df: pd.DataFrame):
    """Creates an st-aggrid interactive table based on a dataframe.

    Args:
        df (pd.DataFrame]): Source dataframe

    Returns:
        dict: The selected row
    """
    options = GridOptionsBuilder.from_dataframe(
        df, enableRowGroup=True, enableValue=True, enablePivot=True
    )

    options.configure_side_bar()

    options.configure_selection(selection_mode='multiple',use_checkbox=True)
    selection = AgGrid(
        df,
        enable_enterprise_modules=True,
        fit_columns_on_grid_load=True,
        gridOptions=options.build(),
        theme="streamlit",
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        allow_unsafe_jscode=True,
    )
    
    return selection


# Vision Globale
def augmented_map_dysfunctions(answers_url, api_url):
    answers = requests.get(answers_url).json()
    data = [item for item in answers]
    
    themes_and_planets_data = requests.get(api_url).json()
    dfs = []
    empty_response_ids = []  # List to track response IDs with empty DataFrames
    
    for response in data:
        answers = response['answers']
        response_id = response['id']
        results = []
        
        for planet in themes_and_planets_data.get('families', []):
            for dysfunction in planet.get('dysfunctions', []):
                for question in dysfunction.get('questions', []):
                    question_id = question['id']
                    matching_answer = next((a for a in answers if a['questionId'] == question_id), None)
                    
                    if matching_answer and matching_answer['answer'] in question.get('responseTrigger', []):
                        results.append({
                            'dysfunction': dysfunction['label'],
                            'weight': dysfunction['weight']
                        })
        
        if results:
            df = pd.DataFrame(results).drop_duplicates(subset=['dysfunction'])
            if not df.empty and 'weight' in df.columns:
                df.rename(columns={'weight': response_id}, inplace=True)
                dfs.append(df)
        else:
            # If no results were found, add the response_id to empty list
            empty_response_ids.append(response_id)
    
    # Create an initial merged DataFrame from valid responses
    if dfs:
        data_merge = reduce(lambda left, right: pd.merge(left, right, on="dysfunction", how="outer"), dfs)
        data_merge.fillna(0, inplace=True)
    else:
        # If no valid responses, start with an empty DataFrame
        data_merge = pd.DataFrame()
    
    # Add empty response IDs as columns with 0.0 values
    for response_id in empty_response_ids:
        data_merge[response_id] = 0.0

    if not data_merge.empty:
        data_merge.drop_duplicates(subset=['dysfunction'], inplace=True)
        data_merge.rename(columns={'dysfunction': ''}, inplace=True)
        return data_merge.set_index('')
    else:
        # Return an empty DataFrame with only empty response IDs as columns
        empty_columns = pd.DataFrame(columns=[''] + empty_response_ids)
        empty_columns.set_index('', inplace=True)
        return empty_columns


# Ecart Reponses
def team_gap(answers_url, api_url):
    """
    Arguments:
    ---------------
    answers_url: address to get the responses from the api.
    api_url: address to get the themes and planets data from the api
    Returns:
    --------------
    A pandas DataFrame where each row represents a family and the sum of dysfunction weights for that family.
    """
    # get answers data from the api 
    answers = requests.get(answers_url)
    all_data = answers.json()
    data = []
    for item in all_data:
        data.append(item)
    
    # get themes and planets data fram the api
    response = requests.get(api_url)
    response.cookies.clear()
    themes_and_planets_data = response.json()
    
    # Initialize a nested dictionary to store the sum of weights for each family and respondent
    family_weights = {}
    # Iterate over each response in the responses list
    for response in data:
        # Extract the respondent ID and their answers
        response_id = response['id']
        answers = response['answers']
        # Iterate over each planet (family) in the themes_and_planets_data
        for planet in themes_and_planets_data['families']:
            family_name = planet['title']
            # Initialize the family dictionary if not already present
            if family_name not in family_weights:
                family_weights[family_name] = {}
            # Initialize the sum of weights for this respondent in this family
            if response_id not in family_weights[family_name]:
                family_weights[family_name][response_id] = 0
            # Iterate over each dysfunction in the current family
            for dysfunction in planet['dysfunctions']:
                for question in dysfunction['questions']:
                    question_id = question['id']
                    # Find the matching answer for the current question
                    matching_answer = next((item for item in answers if item['questionId'] == question_id), None)
                    if matching_answer:
                        user_answer = matching_answer['answer']
                        # Check if the user's answer triggers a dysfunction flag
                        if user_answer in question['responseTrigger']:
                            # Add the dysfunction weight to the family's total for this respondent
                            family_weights[family_name][response_id] += dysfunction['weight']
    # Convert the family_weights dictionary to a DataFrame
    scores_dataframe = pd.DataFrame.from_dict(family_weights, orient="index")
    # Replace NaN values with 0 (for cases where a family has no dysfunctions for some respondents)
    scores_dataframe.fillna(0, inplace=True)
    #calculating standard deviation
    std_list = scores_dataframe.std(axis=1)
    max_list = scores_dataframe.max(axis=1)
    new_std_list = [1 if x==0 else x for x in std_list]
    new_max_list = [1 if x==0 else x for x in max_list]
    agree1 = [x/y for x, y in zip(new_std_list, new_max_list)]
    agree = [100 * item for item in agree1]
    rounded_agree = []
    for item in agree:
        rounded_item = round(item, 1)
        rounded_agree.append(rounded_item)
    #disagree = [100 - item for item in agree]
    gapz = pd.DataFrame(rounded_agree, columns=["% d'accord"])
    #listing families for the final dataframe
    families = ["raison d'être", 'amélioration continue','changement', 'équilibre', 
            'objectif', 'transparence', 'stabilité', 'autonomie', 'collaboration']
    famz = pd.DataFrame(families, columns=['Famille'])
    final = famz.join(gapz)
    final.loc[final["% d'accord"] == 1, "% d'accord"] = 100
    #final.set_index('Famille', inplace=True)
    families = final['Famille'].values.tolist()
    daccord = final["% d'accord"].values.tolist()
    desaccord = []
    for item in daccord:
        desa = round(100-item, 1)
        desaccord.append(desa)
    final["% desaccord"] = desaccord
    
    return final

def category_gauges(dataframe, client_name, results_folder):
    '''This function outputs and saves the gauce graphs for the Ecart des Réponses page
    Each gauge graph is saved with client name and family name
    Arguments:
    ----------
    dataframe (pandas dataframe)
        the dataframe returned by team_gap function
    client_name: (str)
        used in the filename while saving the data
    results_folder: (str)
        path to the folder where the gauge graphs will be saved'''
    
    # Ensure 'dataframe' is a pandas DataFrame
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("The first argument must be a pandas DataFrame.")
    
    # Ensure the necessary columns are present in the dataframe
    required_columns = ["Famille", "% d'accord", "% desaccord"]
    for column in required_columns:
        if column not in dataframe.columns:
            raise ValueError(f"Column '{column}' not found in the dataframe.")
    
    # Iterate over the rows of the dataframe
    for index in range(len(dataframe)):
        try:
            plt.close('all')
            family_value = dataframe.at[index, "% d'accord"]
            anti = dataframe.at[index, "% desaccord"]
            #name = dataframe.at[index, "Famille"].replace(' ', '_')
            
            size = [anti, family_value]
            my_circle = plt.Circle((0, 0), 0.8, color='#EEE5F7')
            
            # Custom wedges
            plt.pie(size, wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
                    colors=['#ffffff', '#8E2DE2'], startangle=270)
            p = plt.gcf()
            p.gca().add_artist(my_circle)
            plt.text(0.001, 0.0001, f'{size[1]}', fontsize=40, color='#283574',
                     horizontalalignment='center', verticalalignment='center')
            plt.tight_layout()
            
            # Save the plot as a PNG file
            plt.savefig(f'{results_folder}/{client_name}_{index}.png', dpi=100, transparent= True)
            plt.clf()
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            

# Top 5 dysfunctions

def dysfunction_frequencies(answers_url, api_url):
    answers = requests.get(answers_url)
    answers.raise_for_status()
    all_data = answers.json()

    response = requests.get(api_url)
    response.cookies.clear()
    themes_and_planets_data = response.json()

    # Create a mapping of dysfunction labels to explanations
    dysfunction_explanations = {
        dysfunction["label"]: dysfunction.get("explanation", "")
        for family in themes_and_planets_data.get("families", [])
        for dysfunction in family.get("dysfunctions", [])
    }

    # Dictionary to store dysfunction weight sums and counts
    dysfunction_scores = defaultdict(lambda: {"weight_sum": 0, "count": 0})

    # Process each user's response
    for response in all_data:
        for answer in response["answers"]:
            for family in themes_and_planets_data.get("families", []):
                for dysfunction in family.get("dysfunctions", []):
                    for question in dysfunction.get("questions", []):
                        if answer["questionId"] == question["id"] and answer["answer"] in question["responseTrigger"]:
                            dysfunction_scores[dysfunction["label"]]["weight_sum"] += dysfunction.get("weight", 0)
                            dysfunction_scores[dysfunction["label"]]["count"] += 1  # Count occurrences

    # Convert dysfunction scores to a DataFrame
    df = pd.DataFrame([
        {
            "dysfunction": label,
            "description": dysfunction_explanations.get(label, "No description available"),
            "weight": (data["weight_sum"] / data["count"]) if data["count"] > 0 else 0  # Normalize by occurrences
        }
        for label, data in dysfunction_scores.items()
    ])

    # Sort dysfunctions by highest average weight and return the top 5
    df = df.sort_values(by="weight", ascending=False).head(5).reset_index(drop=True)

    return df

#Sector ranking
def sector_rank_augmented(assets, answers_url, api_url, dataframe, sector, star):
    '''
    This function is used for comparing client dysfunctions with the sector. A ranking system is used for the comparison: 
    CAUTION: This function uses a helper: dysfunction_frequencies()
    Parameters
    ----------
    csv_assets: str
        Path to where the reference csv's are.
    answers_url: str
        address to get the responses from the api.
    api_url: str
        address to get the themes and planets data from the api
    client_dys : list object
        List of client dysfunctions
    dataframe: pandas dataframe
        returned by the dysfunction_frequencies function
    sector : str
        sector we are comparing the client to. This string will used to fetch the corresponding csv file from assets_path
    star : str
        team, product-vision or product-strategy, also used to fetch the corresponding csv file.
    
    Returns
    -------
    A dataframe with client dysfunctions as they rank on one column and the sector dysfunctions as they rank on the second column.
    '''
    #targets = glob.glob(f'{data_directory}/*.json')
    indus = pd.read_csv(f'{assets}/csv_files/weighted_{star}_{sector}.csv')
    data = dysfunction_frequencies(answers_url, api_url)
    top = dataframe.shape[0]
    client = data.head(top)
    client_dys = client['dysfunction'].values.tolist()
    comp = indus[indus['dysfunction'].isin(client_dys)]
    top_comp = comp.sort_values(by=['impact'], ascending=False)
    
    if top == 1:
        client['rank'] = [100]
        top_comp['rank'] = [100]
    elif top == 2:
        client['rank'] = [100, 66]
        top_comp['rank'] = [100, 66]
    elif top == 3 :
        client['rank'] = [100, 66, 33]
        top_comp['rank'] = [100, 66, 33]
    elif top == 4:
        client['rank'] = [100, 80, 60, 40]
        top_comp['rank'] = [100, 80, 60, 40]
    else:
        client['rank'] = [100, 80, 60, 40, 20]
        top_comp['rank'] = [100, 80, 60, 40, 20]
    comp2 = top_comp.drop(columns=['impact'])
    final = client.merge(comp2, on='dysfunction', suffixes=('_vous', '_secteur'))
    final.drop(columns=['Unnamed: 0'], inplace=True)
    return final

# cas marge

def cas_marge(answers_url, api_url):
    '''Returns marginal cases as dataframe
    CAUTION: uses helper function (augmented_map_dysfunctions)
    Parameters
    ----------
    answers_url: address to get the responses from the api.
    api_url: address to get the themes and planets data from the api
    '''
    result1 = augmented_map_dysfunctions(answers_url, api_url)
    people = len(result1.columns.tolist())
    max_elements = result1.max(axis=1)
    result1['total'] = result1.sum(axis=1)
    #result1.drop(columns = ['vous', 'vous_drop'], inplace = True)
    result1['weight'] = max_elements
    sub0 = result1.loc[result1['total'] == result1['weight']]
    sub1 = result1.loc[result1['total'] == result1['weight'].values * people - result1['weight'].values]
    result = pd.concat([sub0, sub1])
    #result['identifié par'] = np.where(result['total'] == result['weight'], 1, len(targets)-1)
    result.drop(columns=['total', 'weight'], inplace=True)
    result[result != 0] = 1
    return result

# solutions
def get_all_solutions(assets_path, answers_url, api_url):
    '''returns the dataframe needed for the solutions page of e-diag report
    CAUTION: uses helper function dysfunction_frequencies
    Arguments:
    ---------
    assets_path: str
    answers_url: str
        address to get the responses from the api.
    api_url: str
        address to get the themes and planets data from the api
        Path to the json file with themes and planets data.
    '''
    # get top5 dysfunctions
    frequencies = dysfunction_frequencies(answers_url, api_url)
    top5 = frequencies.head(5)
    client_dys = top5['dysfunction'].values
    # read the solutions dataframe
    leveled_solutions = pd.read_csv(f'{assets_path}/csv_files/leveled_solutions_multiverse_20250312.csv')
    # select the solutions that concern the top5 dysfunctions
    relevent = leveled_solutions[leveled_solutions['dysfunction'].isin(top5['dysfunction'])]
    return relevent

def get_solutions(assets_path, answers_url, api_url):
    '''returns the dataframe needed for the solutions page of e-diag report
    CAUTION: uses helper function dysfunction_frequencies
    Arguments:
    ---------
    assets_path: str
    answers_url: str
        address to get the responses from the api.
    api_url: str
        address to get the themes and planets data from the api
        Path to the json file with themes and planets data.
    '''
    # get top5 dysfunctions
    frequencies = dysfunction_frequencies(answers_url, api_url)
    top5 = frequencies.head(5)
    client_dys = top5['dysfunction'].values
    # read the solutions dataframe
    leveled_solutions = pd.read_csv(f'{assets_path}/csv_files/leveled_solutions_multiverse.csv')
    # select the solutions that concern the top5 dysfunctions
    relevent = leveled_solutions[leveled_solutions['dysfunction'].isin(top5['dysfunction'])]
    # select maturity levels
    locs = []
    for i in range(len(client_dys)): 
        loc = np.where((relevent.dysfunction == client_dys[i]))
        locs.append(loc)
    dfs = []
    for i in range(len(locs)):
        df = relevent.iloc[locs[i]]
        dfs.append(df)

    selected_sols = pd.concat(dfs)
    return selected_sols

def get_solutions_nextoo(assets_path, answers_url, api_url):
    '''returns the dataframe needed for the NEXTOO solutions page of e-diag report
    CAUTION: uses helper function dysfunction_frequencies
    Arguments:
    ---------
    assets_path: str
    answers_url: str
        address to get the responses from the api.
    api_url: str
        address to get the themes and planets data from the api
        Path to the json file with themes and planets data.
    '''
    # get top5 dysfunctions
    frequencies = dysfunction_frequencies(answers_url, api_url)
    top5 = frequencies.head(5)
    client_dys = top5['dysfunction'].values
    # read the solutions dataframe
    leveled_solutions = pd.read_csv(f'{assets_path}/csv_files/leveled_solutions_multiverse.csv')
    # select the solutions that concern the top5 dysfunctions
    relevent = leveled_solutions[leveled_solutions['dysfunction'].isin(top5['dysfunction'])]
    # select maturity levels
    locs = []
    for i in range(len(client_dys)):
        loc = np.where((relevent.dysfunction == client_dys[i]) & (relevent.Maturity == 'Nextoo'))
        locs.append(loc)
        loc1 = np.where((relevent.dysfunction == client_dys[i]) & (relevent.Maturity == 'Niveau 1'))
        locs.append(loc1)
    dfs = []
    for i in range(len(locs)):
        df = relevent.iloc[locs[i]]
        dfs.append(df)
        
    selected_sols = pd.concat(dfs)
    df = selected_sols.dropna(axis=0, subset=['Solutions'])
    # Sort the dataframe so that rows with "Nextoo" appear first
    df_sorted = df.sort_values(by="Maturity", ascending=False, key=lambda col: col == "Nextoo")
    # Drop duplicates, keeping the first occurrence
    df_unique = df_sorted.drop_duplicates(subset="dysfunction", keep="first").reset_index(drop=True)
    
    return df_unique

def solutions_page_logo(assets, dataframe, results_folder, client_name):
    """
    Creates the solutions pdf page by writing the dysfunction, solution, and impact strings 
    on the template pdf page, including a logo if "Maturity" is "Nextoo".
    
    Arguments:
    ----------
    assets: path where the template pdf is located.
    dataframe: DataFrame returned by the get_solutions function.
    results_folder: path to save the page.
    client_name: used for naming the page.
    """
    # Open the PDF file
    pdf_document = fitz.open(f'{assets}/pdf_templates/Solutions1.pdf')
    page_number = 0
    page = pdf_document.load_page(page_number)
    
    # Insert fonts
    font_file = f'{assets}/font_files/AzoSans-Black.ttf'
    page.insert_font(fontfile=font_file, fontname="AzoBold")
    font_file2 = f'{assets}/font_files/AzoSans-Light.ttf'
    page.insert_font(fontfile=font_file2, fontname="AzoLight")
    
    # Paths and positions
    logo_path = f'{assets}/images/hacoeur.png'  # Path to Nextoo logo
    logo_width, logo_height = 45, 45  # Logo dimensions
    normal_positions = [(90, 170, 450, 500), (460, 170, 810, 500), (90, 400, 700, 800)]
    nextoo_positions = [(130, 170, 450, 500), (500, 170, 810, 500), (130, 400, 700, 800)]
    solution_positions = [(90, 215), (460, 215), (90, 445)]  # Adjusted for logo
    impact_positions = [(90, 230, 450, 700), (460, 230, 810, 700), (90, 450, 810, 700)]
    logo_positions = [(85, 170), (455, 170), (85, 400)]  # Logo positions
    orange = fitz.pdfcolor["orange"]
    
    # Iterate over dysfunctions
    number_of_dysfunctions = dataframe.shape[0]
    max_items = min(number_of_dysfunctions, 3)
    
    for i in range(max_items):
        dys = dataframe['dysfunction'].iloc[i]
        sol = dataframe['Solutions'].iloc[i]
        imp = dataframe['Impacts'].iloc[i]
        maturity = dataframe['Maturity'].iloc[i]
        
        # If maturity is "Nextoo", add logo and adjust text positions
        if 'Nextoo' in maturity:
            # Insert dysfunction text
            page.insert_textbox(nextoo_positions[i], dys, fontsize=14, fontname="AzoBold", lineheight=1.5)
            x, y = logo_positions[i]
            page.insert_image(fitz.Rect(x, y, x + logo_width, y + logo_height), filename=logo_path)
            # Adjust positions for text to the right of the logo
            sol_position = (solution_positions[i][0] + 40, solution_positions[i][1])  # Shift by 40px
            page.insert_text(sol_position, sol, fontsize=12, fontname="AzoBold", color=orange, lineheight=1.5)
        else:
            # Insert dysfunction text
            page.insert_textbox(normal_positions[i], dys, fontsize=14, fontname="AzoBold", lineheight=1.5)
            # Insert solution text at the default position
            page.insert_text(solution_positions[i], sol, fontsize=12, fontname="AzoBold", color=orange, lineheight=1.5)
        
        # Insert impact text
        page.insert_textbox(impact_positions[i], imp, fontsize=12, fontname="AzoLight", lineheight=1.5)
    
    # Save the modified PDF to a new file
    pdf_document.save(f'{results_folder}/{client_name}_solutions_page1.pdf')
    pdf_document.close()

def solutions_page_logo2(assets, dataframe, results_folder, client_name):
    """
    Creates the solutions pdf page by writing the dysfunction, solution, and impact strings 
    on the template pdf page, including a logo if "Maturity" is "Nextoo".
    
    Arguments:
    ----------
    assets: path where the template pdf is located.
    dataframe: DataFrame returned by the get_solutions function.
    results_folder: path to save the page.
    client_name: used for naming the page.
    """
    # Open the PDF file
    pdf_document = fitz.open(f'{assets}/pdf_templates/Solutions2.pdf')
    page_number = 0
    page = pdf_document.load_page(page_number)
    
    # Insert fonts
    font_file = f'{assets}/font_files/AzoSans-Black.ttf'
    page.insert_font(fontfile=font_file, fontname="AzoBold")
    font_file2 = f'{assets}/font_files/AzoSans-Light.ttf'
    page.insert_font(fontfile=font_file2, fontname="AzoLight")
    
    # Paths and positions
    logo_path = f'{assets}/images/hacoeur.png'  # Path to Nextoo logo
    logo_width, logo_height = 45, 45  # Logo dimensions
    normal_positions = [(90, 250, 445, 500), (460, 250, 810, 700)]
    nextoo_positions = [(130, 250, 445, 500), (500, 250, 810, 700)]
    solution_positions = [(90, 300), (460, 300)]
    impact_positions = [(95, 320, 450, 700), (465, 320, 810, 700)]
    logo_positions = [(85, 250), (455, 250)]  # Logo positions
    orange = fitz.pdfcolor["orange"]

    # Slice the dataframe for the last two rows
    dataframe2 = dataframe.iloc[-2:]  # Select the last 2 rows
    dataframe2.reset_index(inplace=True, drop=True)
    number_of_dysfunctions = dataframe2.shape[0]  # Count the rows in the sliced dataframe
    max_items = min(number_of_dysfunctions, len(normal_positions))  # Ensure within bounds
    
    for i in range(max_items):
        dys = dataframe2['dysfunction'].iloc[i]
        sol = dataframe2['Solutions'].iloc[i]
        imp = dataframe2['Impacts'].iloc[i]
        maturity = dataframe2['Maturity'].iloc[i]
        
        # If maturity is "Nextoo", add logo and adjust text positions
        if "Nextoo" in maturity:
            # Insert dysfunction text
            page.insert_textbox(nextoo_positions[i], dys, fontsize=14, fontname="AzoBold", lineheight=1.5)
            x, y = logo_positions[i]
            page.insert_image(fitz.Rect(x, y, x + logo_width, y + logo_height), filename=logo_path)
            # Adjust positions for text to the right of the logo
            sol_position = (solution_positions[i][0] + 40, solution_positions[i][1])  # Shift by 40px
            page.insert_text(sol_position, sol, fontsize=12, fontname="AzoBold", color=orange, lineheight=1.5)
        else:
            # Insert dysfunction text
            page.insert_textbox(normal_positions[i], dys, fontsize=14, fontname="AzoBold", lineheight=1.5)
            # Insert solution text at the default position
            page.insert_text(solution_positions[i], sol, fontsize=12, fontname="AzoBold", color=orange, lineheight=1.5)
        
        # Insert impact text
        page.insert_textbox(impact_positions[i], imp, fontsize=12, fontname="AzoLight", lineheight=1.5)
    
    # Save the modified PDF to a new file
    pdf_document.save(f'{results_folder}/{client_name}_solutions_page2.pdf')
    pdf_document.close()

# DISAGREEMENTS
def augmented_disagree(answers_url, api_url):
    '''Returns the disagreements (50% of the team identifies the dysfunction)
    as a list
    CAUTION: uses helper function augmented_map_dysfunctions 
    Arguments:
    ----------
    answers_url: str
        address to get the responses from the api.
    api_url: str
        address to get the themes and planets data from the api
    '''
    
    result3 = augmented_map_dysfunctions(answers_url, api_url)
    people = result3.shape[1]
    #convert all non zero to 1 all zeros to 0
    result3[result3 != 0] = 1
    result3['sum'] = result3.sum(axis=1)
    result3['verdict'] = round(result3['sum']/people, 1)
    #df = df[(df['col'] < -0.25) | (df['col'] > 0.25)]
    verdicts = result3[(result3['verdict'] < 0.6) & (result3['verdict'] > 0.4)]
    #result3.loc[result3["verdict"]==0.5, "verdict"] = 'disagree'
    #verdicts.reset_index(inplace=True)
    #dis = result3.loc[result3['verdict'] == 'disagree']
    #disagreements = verdicts[''].tolist()
    
    return verdicts

# SECTOR OVERLAY
def average_family_scores(answers_url, api_url):
    """
    Arguments:
    ---------------
    answers_url: address to get the responses from the api.
    api_url: address to get the themes and planets data from the api
    Returns:
    --------------
    a pandas dataframe with average scores of each dysfunction families
    """
    # Load the single JSON file containing responses and themes/planets data
    # get answers data from the api 
    answers = requests.get(answers_url)
    all_data = answers.json()
    data = []
    for item in all_data:
        data.append(item)
    
    # get themes and planets data fram the api
    response = requests.get(api_url)
    response.cookies.clear()
    themes_and_planets_data = response.json()
    
    # Initialize a nested dictionary to store the sum of weights for each family and respondent
    family_weights = {}
    # Iterate over each response in the responses list
    for response in data:
        # Extract the respondent ID and their answers
        response_id = response['id']
        answers = response['answers']
        # Iterate over each planet (family) in the themes_and_planets_data
        for planet in themes_and_planets_data['families']:
            family_name = planet['title']
            # Initialize the family dictionary if not already present
            if family_name not in family_weights:
                family_weights[family_name] = {}
            # Initialize the sum of weights for this respondent in this family
            if response_id not in family_weights[family_name]:
                family_weights[family_name][response_id] = 0
            # Iterate over each dysfunction in the current family
            for dysfunction in planet['dysfunctions']:
                for question in dysfunction['questions']:
                    question_id = question['id']
                    # Find the matching answer for the current question
                    matching_answer = next((item for item in answers if item['questionId'] == question_id), None)
                    if matching_answer:
                        user_answer = matching_answer['answer']
                        # Check if the user's answer triggers a dysfunction flag
                        if user_answer in question['responseTrigger']:
                            # Add the dysfunction weight to the family's total for this respondent
                            family_weights[family_name][response_id] += dysfunction['weight']
    # Convert the family_weights dictionary to a DataFrame
    scores_dataframe = pd.DataFrame.from_dict(family_weights, orient="index")
    # Replace NaN values with 0 (for cases where a family has no dysfunctions for some respondents)
    scores_dataframe.fillna(0, inplace=True)
    df = scores_dataframe.reset_index(level=0)
    people = df.shape[1]
    df['score'] = df.sum(axis=1, numeric_only=True)/people
    final = df[['index', 'score']].copy() 
    return final

def sector_rank_page(assets, dataframe, results_folder, client_name, points_forts, points_faibles):
    """
    Creates the sector comparison pdf page by creating the writing the dysfunction strings on the
    template pdf page.
    Arguments:
    ----------
    assets: path where the template pdf is located.
    dataframe: returned by the sector_rank_augmeted()
    results_folder: path to save the page and to fetch the gauche png images
    client_name: used for naming the page
    """
    # Open the PDF file
    pdf_document = fitz.open(f'{assets}/pdf_templates/emptyPage.pdf')
    
    # Define the page to insert the image (0-based index)
    page_number = 0
    page = pdf_document.load_page(page_number)
    font_file= f'{assets}/font_files/AzoSans-Black.ttf'
    page.insert_font(fontfile=font_file, fontname="Azo")
    
    font_file2= f'{assets}/font_files/AzoSans-Light.ttf'
    page.insert_font(fontfile=font_file2, fontname="AzoLight")
    
    # title
    
    title = "Votre positionnement"
    title_position = (75, 80)
    page.insert_text(title_position, title, fontsize= 14, fontname="Azo")
    
    ### IMAGES ###
    positions = [(70,25),(310,25),(550,25),(190,185),(425,185)]
    number_of_dysfunctions = dataframe.shape[0]
    for i in range(number_of_dysfunctions):
        image_rect = fitz.Rect(positions[i][0], positions[i][1], positions[i][0] + 200, positions[i][1] + 250)
        page.insert_image(image_rect, filename=f'{results_folder}/{client_name}_dys{i}_raw.png')
        
    ### TEXTS ####
    rectangles = [(70, 200, 300, 500), (300, 200, 550, 600), (552, 200, 800, 700), (190, 360, 400, 450),(430, 360, 660, 600)]
    for i in range(number_of_dysfunctions):
        rect = rectangles[i]
        text = dataframe['dysfunction'].iloc[i]
        page.insert_textbox(rect, text, fontsize=11, fontname="Azo", lineheight=1.5)
    
    # Legende images
    position1 = (50,410)
    image_rect1 = fitz.Rect(position1[0], position1[1], position1[0] + 13, position1[1] + 13)
    page.insert_image(image_rect1, filename=f"{assets}/images/blue_square.png")
    
    position2 = (120,410)
    image_rect2 = fitz.Rect(position2[0], position2[1], position2[0] + 13, position2[1] + 13)
    page.insert_image(image_rect2, filename=f"{assets}/images/orange_square.png")
    
    # Legende texts
    text = "Vous"
    p = (68, 420)
    page.insert_text(p, text, fontsize=10, fontname="AzoLight")
    text2 = "Votre secteur d'activité"
    p2 = (138, 420)
    page.insert_text(p2, text2, fontsize=10, fontname="AzoLight")
    
    # Points Forts
    points_forts_title = "Points forts : "
    p3 = (50, 460)
    page.insert_text(p3, points_forts_title, fontsize=12, fontname= "Azo")
    points_forts_list = '; '.join(points_forts)
    #points_forts_text = f"Les dysfonctionnements suivants vous affectent moins par rapport au reste de votre sécteur d'activité: {points_forts_list}"
    p_list1 = (135, 450, 810, 800)
    page.insert_textbox(p_list1, points_forts_list, fontsize=12, fontname="AzoLight", lineheight=1.5)
    
    # Points Faibles
    points_faibles_title = "Points faibles : "
    p4 = (50, 510)
    page.insert_text(p4, points_faibles_title, fontsize=12, fontname= "Azo")
    points_faibles_list = '; '.join(points_faibles)
    #points_faibles_text = f"Les dysfonctionnements suivants vous affectent moins par rapport au reste de votre sécteur d'activité: {points_faibles_list}"
    p_list2 = (140, 500, 810, 800)
    page.insert_textbox(p_list2, points_faibles_list, fontsize=12, fontname="AzoLight", lineheight=1.5)

    
    # Save the modified PDF to a new file
    pdf_document.save(f'{results_folder}/{client_name}_ranks.pdf')
    pdf_document.close()

def ecart_team_page1(assets, results_folder, client_name, agree, ecart):
    """
    Insert an image into a PDF file at a specified position.

    Args:
    pdf_path (str): Path to the existing PDF file.
    image_path (str): Path to the image file to insert.
    output_pdf_path (str): Path to save the modified PDF file.
    position (tuple): (x, y) coordinates for the image insertion.
    """
    # Open the PDF file
    pdf_document = fitz.open(f'{assets}/pdf_templates/Ecart_template1.pdf')
    
    # Define the page to insert the image (0-based index)
    page_number = 0
    page = pdf_document.load_page(page_number)
    font_file= f'{assets}/font_files/AzoSans-Light.ttf'
    page.insert_font(fontfile=font_file, fontname="Azo")

    # Horizontal bar
    position = (75,40)
    image_rect = fitz.Rect(position[0], position[1], position[0] + 700, position[1] + 200)
    page.insert_image(image_rect, filename=f'{results_folder}/{client_name}_ecart_figure.png')
    
    text = f"""Il y a un écart dans {ecart} % des réponses aux questions. Ainsi, l'équipe est en phase à {agree} % ."""
    text2 = "Chaque dysfonctionnement appartient à une famille qui représente un axe principal du thème collaborer en équipe :"
                 
    # Insert text
    p = (100, 190)
    page.insert_text(p, text, fontsize=12, fontname="Azo")
    
    # Insert text2
    p = (100, 210)
    page.insert_text(p, text2, fontsize=12, fontname="Azo")
    
    # raison d'être
    position1 = (65,242)
    image_rect1 = fitz.Rect(position1[0], position1[1], position1[0] + 90, position1[1] + 90)
    page.insert_image(image_rect1, filename=f"{results_folder}/{client_name}_0.png")
    
    # amélioration
    position2 = (410,242)
    image_rect2 = fitz.Rect(position2[0], position2[1], position2[0] + 90, position2[1] + 90)
    page.insert_image(image_rect2, filename=f'{results_folder}/{client_name}_1.png')
    # changement
    position3 = (65,387)
    image_rect3 = fitz.Rect(position3[0], position3[1], position3[0] + 90, position3[1] + 90)
    page.insert_image(image_rect3, filename=f'{results_folder}/{client_name}_2.png')
    # équilibre
    position4 = (410,387)
    image_rect4 = fitz.Rect(position4[0], position4[1], position4[0] + 90, position4[1] + 90)
    page.insert_image(image_rect4, filename=f'{results_folder}/{client_name}_3.png')

    # Save the modified PDF to a new file
    pdf_document.save(f'{results_folder}/{client_name}_ecart_reponses1.pdf')
    pdf_document.close()

def ecart_team_page2(assets, results_folder, client_name):
    """
    Creates the ecart des réponses pdf page 2 for the report.

    Args:
    pdf_path (str): Path to the existing PDF file.
    image_path (str): Path to the image file to insert.
    output_pdf_path (str): Path to save the modified PDF file.
    position (tuple): (x, y) coordinates for the image insertion.
    """
    # Open the PDF file
    pdf_document = fitz.open(f'{assets}/pdf_templates/Ecart_template2.pdf')
    
    # Define the page to insert the image (0-based index)
    page_number = 0
    page = pdf_document.load_page(page_number)
    #font_file= f'{assets}/AzoSans-Light.ttf'
    #page.insert_font(fontfile=font_file, fontname="Azo")
    
    # objectif
    position1 = (64,100)
    image_rect1 = fitz.Rect(position1[0], position1[1], position1[0] + 90, position1[1] + 90)
    page.insert_image(image_rect1, filename=f'{results_folder}/{client_name}_4.png')
    
    # transparence
    position2 = (410,100)
    image_rect2 = fitz.Rect(position2[0], position2[1], position2[0] + 90, position2[1] + 90)
    page.insert_image(image_rect2, filename=f'{results_folder}/{client_name}_5.png')
    # stabilite
    position3 = (64,250)
    image_rect3 = fitz.Rect(position3[0], position3[1], position3[0] + 90, position3[1] + 90)
    page.insert_image(image_rect3, filename=f'{results_folder}/{client_name}_6.png')
    # autonomie
    position4 = (410,250)
    image_rect4 = fitz.Rect(position4[0], position4[1], position4[0] + 90, position4[1] + 90)
    page.insert_image(image_rect4, filename=f'{results_folder}/{client_name}_7.png')
    # collaboration
    position5 = (64,400)
    image_rect5 = fitz.Rect(position5[0], position5[1], position5[0] + 90, position5[1] + 90)
    page.insert_image(image_rect5, filename=f'{results_folder}/{client_name}_8.png')

    # Save the modified PDF to a new file
    pdf_document.save(f'{results_folder}/{client_name}_ecart_reponses2.pdf')
    pdf_document.close()

def merger(pdf_list, client_name, results_folder, page_name):
    result = fitz.open()
    for pdf in pdf_list:
        with fitz.open(pdf) as mfile:
            result.insert_pdf(mfile)
    result.save(f"{results_folder}/{client_name}_{page_name}.pdf", deflate=True)
    result.close()

##################################    
###### USER INTERFACE ############
##################################

st.title("Générateur d'image pour e-diag augmenté")

# input paths
st.header('Input paths')

here = os.getcwd()
assets = os.path.join(here, 'assets')
#st.write(f'assets files fetched from: {assets}')
#path to save results
results_folder = os.path.join(here, 'results')
#st.write(f'output files and report saved to: {results_folder}')

single = st.checkbox("Cet e-diag ne concerne qu'une seule personne")

if single:
    st.write("Une seule personne a répondu. 'Ecart des réponses', 'Cas en marge' et 'points de désaccord' non disponibles.")

# naming convention for files
st.header('input information')
client_real_name = st.text_input("Le nom de l'équipe concerné : ")
client_name = client_real_name.replace(' ','_')
st.write(f'save name is {client_name}')

form_theme = 'Collaborer en Équipe'
api_url = st.secrets["api_url"]
thematics = 'team'

campaign = st.text_input('Enter campaign id')
answers_prefix = st.secrets["answers_prefix"]
answers_url = f'{answers_prefix}/{campaign}'

st.header("Vision Globale")

everyone = augmented_map_dysfunctions(answers_url, api_url)
if single:
    fig_everyone, ax_everyone = plt.subplots(figsize=(10,4)) 
else:
    fig_everyone, ax_everyone = plt.subplots(figsize=(10,10))
sns.heatmap(everyone,annot=False, linewidths=.5, ax=ax_everyone, xticklabels=False, cbar=False, cmap="BuPu")
plt.tight_layout()
st.pyplot(fig_everyone)
# save heatmap
# fig_everyone.savefig(f'{results_folder}/{client_name}_vision_globale.png', dpi=100, transparent=True)
#st.write(f'created: {client_name}_vision_globale.png ✅')

st.header('Écart des réponses')
if single == False:
    team_gap_df = team_gap(answers_url, api_url)
    # create gauge images 
    category_gauges(team_gap_df, client_name, results_folder)
    # create the bar plot for écart réponses page
    # Create figure and axis
    ecart_fig, ecart_ax = plt.subplots(figsize=(10, 0.5))
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    agree = round(team_gap_df["% d'accord"].mean())
    ecart = 100-agree

    # Data
    categories = ['']  # Single bar, so single category
    percentages = [agree, ecart]  # The lengths of the segments of the bar
    colors = ['#FF5F6D', '#4A00E0']  # The colors for each segment
        
    # Create the stacked bar graph
    bars = plt.barh(categories, percentages[0], color=colors[0])
    bars = plt.barh(categories, percentages[1], left=percentages[0], color=colors[1])
                
    # Annotate the segments of the stacked bar
    plt.text(percentages[0] / 2, 0, f'{percentages[0]}%', va='center', ha='center', color='white', fontweight='bold')
    plt.text(percentages[0] + percentages[1] / 2, 0, f'{percentages[1]}%', va='center', ha='center', color='white', fontweight='bold')
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    plt.axis('off')
    # Remove the spines
    ecart_ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.write(f"Il y a un écart dans {ecart}% des réponses aux questions. Ainsi, l'équipe est en phase à {agree}%")
    st.pyplot(ecart_fig)
else:
    st.write("Une seule personne a répondu, les cas en marge ne sont pas disponibles")

st.header("Les cas en marge")
if single == False:
    # cas en marge
    cas_marge_df = cas_marge(answers_url, api_url)
    #show cas marge
    st.dataframe(data=cas_marge_df)
else:
    st.write("Une seule personne a répondu, les points de désaccord ne sont pas disponibles")

st.header('Les point de désaccord')
if single == False:
    # cas en marge
    disagree_df = augmented_disagree(answers_url, api_url)
    # show disagree
    st.dataframe(data=disagree_df)

st.header("Solutions")
solutions = get_all_solutions(assets, answers_url, api_url)
selection = aggrid_interactive_table(df=solutions)
filtered_solutions = selection['selected_rows']
st.write("selected solutions")
st.dataframe(data=filtered_solutions)

