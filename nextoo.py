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
import requests
from collections import defaultdict
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode
from itertools import count

###############################
########## FUNCTIONS ##########
###############################
 
st.set_page_config(layout='wide', page_icon=':octopus:')

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


# Most serious dysfunctions
def most_serious(answers_url, api_url):
    answers = requests.get(answers_url)
    answers.raise_for_status()
    all_data = answers.json()

    response = requests.get(api_url)
    response.cookies.clear()
    themes_and_planets_data = response.json()

    # Filter out the people who turbo-clicked
    unfiltered_data = [item for item in all_data]

    all_data = [respondent for respondent in unfiltered_data if len(set(ans["answer"] for ans in respondent["answers"])) > 1]

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
            "weight": (data["weight_sum"] / data["count"]) if data["count"] > 0 else 0,  # Normalize by occurrences
            "people detected": data["count"]
        }
        for label, data in dysfunction_scores.items()
    ])

    # Sort dysfunctions by highest average weight and return the top 5
    df = df.sort_values(by="weight", ascending=False).head(5).reset_index(drop=True)

    return df

# Vision Globale
def augmented_map_dysfunctions(answers_url, api_url):
    response = requests.get(answers_url)
    response.raise_for_status()
    all_data = response.json()

    # Filter out the people who turbo-clicked
    unfiltered_data = [item for item in all_data]

    data = [respondent for respondent in unfiltered_data if len(set(ans["answer"] for ans in respondent["answers"])) > 1]
    
    # Fetch themes and planets data
    response = requests.get(api_url)
    response.raise_for_status()
    response.cookies.clear()
    themes_and_planets_data = response.json()
    
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


# Top 5 dysfunctions
def dysfunction_frequencies(answers_url, api_url):
    # Fetch the responses from the answers API
    response = requests.get(answers_url)
    response.raise_for_status()
    all_data = response.json()

    # Filter out the people who turbo-clicked
    unfiltered_data = [item for item in all_data]

    data = [respondent for respondent in unfiltered_data if len(set(ans["answer"] for ans in respondent["answers"])) > 1]

    # Fetch themes and planets data
    response = requests.get(api_url)
    response.raise_for_status()
    response.cookies.clear()
    themes_and_planets_data = response.json()

    # Create a dictionary to map dysfunction labels to their explanations
    dysfunction_explanations = {}
    for family in themes_and_planets_data.get('families', []):
        for dysfunction in family.get('dysfunctions', []):
            dysfunction_label = dysfunction.get('label', 'Unknown Dysfunction')
            dysfunction_explanations[dysfunction_label] = dysfunction.get('explanation', '')

    # List to store DataFrames for each response
    dfs = []

    # Process each user's response
    for response in data:
        answers = response['answers']
        response_id = response['id']
        results = []

        # Iterate over each family and dysfunction
        for family in themes_and_planets_data.get('families', []):
            for dysfunction in family.get('dysfunctions', []):
                for question in dysfunction.get('questions', []):
                    question_id = question['id']
                    matching_answer = next((item for item in answers if item['questionId'] == question_id), None)

                    if matching_answer:
                        user_answer = matching_answer['answer']
                        if user_answer in question.get('responseTrigger', []):
                            results.append({
                                'dysfunction': dysfunction.get('label', 'Unknown Dysfunction'),
                                'weight': dysfunction.get('weight', 0),
                                'family': family.get('title', 'Unknown Family'),
                                'explanation': dysfunction.get('explanation', '')  # Fallback from API
                            })

        # Create a DataFrame for the current response
        df = pd.DataFrame(results)
        if not df.empty:
            df.drop_duplicates(subset=['dysfunction'], inplace=True)
            df.rename(columns={'weight': response_id}, inplace=True)
            dfs.append(df)

    if not dfs:
        return pd.DataFrame(columns=['dysfunction', 'description', 'av_weight'])

    # Merge all DataFrames on the 'dysfunction' column
    for i, df in enumerate(dfs):
        df.rename(columns={col: f"{col}_{i}" for col in df.columns if col != "dysfunction"}, inplace=True)

    data_merge = reduce(lambda left, right: pd.merge(left, right, on="dysfunction", how="outer"), dfs)
    
    data_merge.fillna(0, inplace=True)

    # Calculate average weight
    numeric_cols = data_merge.select_dtypes(include=np.number).columns
    data_merge['av_weight'] = data_merge[numeric_cols].mean(axis=1)

    # Add explanations from the dictionary if they are missing
    data_merge['description'] = data_merge['dysfunction'].map(dysfunction_explanations)

    # Create the final DataFrame with needed columns
    final_df = data_merge[['dysfunction', 'description', 'av_weight']].copy()

    # Sort by average weight and reset the index
    final_df.sort_values(by='av_weight', ascending=False, inplace=True)
    final_df.reset_index(drop=True, inplace=True)

    return final_df


# Anti-dysfunctions
def points_forts(answers_url, api_url):
    # Fetch the responses from the answers API
    response = requests.get(answers_url)
    response.raise_for_status()
    all_data = response.json()

    # Filter out the people who turbo-clicked
    unfiltered_data = [item for item in all_data]

    data = [respondent for respondent in unfiltered_data if len(set(ans["answer"] for ans in respondent["answers"])) > 1]

    # Fetch themes and planets data
    response = requests.get(api_url)
    response.raise_for_status()
    response.cookies.clear()
    themes_and_planets_data = response.json()

    # Create a dictionary to map dysfunction labels to their explanations
    dysfunction_explanations = {}
    for family in themes_and_planets_data.get('families', []):
        for dysfunction in family.get('dysfunctions', []):
            dysfunction_label = dysfunction.get('label', 'Unknown Dysfunction')
            dysfunction_explanations[dysfunction_label] = dysfunction.get('explanation', '')

    # List to store DataFrames for each response
    dfs = []

    diagnosed_df = dysfunction_frequencies(answers_url, api_url)
    diagnosed = diagnosed_df["dysfunction"].values.tolist()

    # Process each user's response
    for response in data:
        answers = response['answers']
        results = []

        # Iterate over each family and dysfunction
        for family in themes_and_planets_data.get('families', []):
            for dysfunction in family.get('dysfunctions', []):
                for question in dysfunction.get('questions', []):
                    question_id = question['id']
                    matching_answer = next((item for item in answers if item['questionId'] == question_id), None)

                    if matching_answer:
                        user_answer = matching_answer['answer']
                        if user_answer not in question.get('responseTrigger', []):
                            results.append({
                                'dysfunction': dysfunction.get('label', 'Unknown Dysfunction'),
                                'weight': dysfunction.get('weight', 0)
                            })

        # Create a DataFrame for the current response
        df = pd.DataFrame(results)
        if not df.empty:
            df.drop_duplicates(subset=['dysfunction'], inplace=True)
            dfs.append(df)

    if not dfs:
        return pd.DataFrame(columns=['dysfunction', 'weight'])

    data_merge = pd.concat(dfs).drop_duplicates(subset=['dysfunction']).reset_index(drop=True)
    
    result1 = data_merge.sort_values(by='weight', ascending=False)
    result1.reset_index(drop=True, inplace=True)
    result3 = result1[~result1['dysfunction'].isin(diagnosed)]
    result3.rename(columns={"dysfunction":"dysfonctionnement"}, inplace=True)

    return result3


def vision_globale_page(assets_path, client_name, results_folder):
    """Creates the Vision Globale pdf page
    Arguments:
    ----------
    assets_path: str
        path to the pdf templates
    client_name: str
        naming convention for the figures
    results_folder: str
        path to fetch the vison globale heatmap figure"""
    
    pdf_document = fitz.open(f'{assets_path}/pdf_templates/emptyPage.pdf')
    page_number = 0
    page = pdf_document.load_page(page_number)
    
    light_font_file= f'{assets_path}/font_files/AzoSans-Light.ttf'
    bold_font_file= f'{assets_path}/font_files/AzoSans-Black.ttf'
    page.insert_font(fontfile=light_font_file, fontname="AzoLight")
    page.insert_font(fontfile=bold_font_file, fontname="AzoBold")
    
    title = "Vision Globale"
    title_position = (100, 80)
    page.insert_text(title_position, title, fontsize= 14, fontname="AzoBold")
    
    #insert image
    position1 = (160,70)
    image_rect1 = fitz.Rect(position1[0], position1[1], position1[0] + 500, position1[1] + 470)
    page.insert_image(image_rect1, filename=f"{results_folder}/{client_name}_vision_globale.png")
    
    # Insert Legende
    legende_position = (550,100)
    legende = fitz.Rect(legende_position[0], legende_position[1], legende_position[0] + 400, legende_position[1] + 400)
    page.insert_image(legende, filename=f"{assets_path}/images/legendeHeatmap.png")
    
    text = """Légende :\n* Plus le rectangle est violet foncé, plus le dysfonctionnement est impactant d'après la dysfonctiothèque.\n* Chaque colonne correspond à une personne de l’équipe. """
    p = (90, 540, 810, 700)
    page.insert_textbox(p, text, fontsize=8, fontname="AzoLight", lineheight=1.5)

    # Save the modified PDF to a new file
    pdf_document.save(f'{results_folder}/{client_name}_visionglobale.pdf')
    pdf_document.close()

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
    top5 = dysfunction_frequencies(answers_url, api_url)
    client_dys = top5['dysfunction'].values
    # read the solutions dataframe
    leveled_solutions = pd.read_csv(f'{assets_path}/csv_files/leveled_solutions_multiverse_20250312.csv')
    # select the solutions that concern the top5 dysfunctions
    relevent = leveled_solutions[leveled_solutions['dysfunction'].isin(top5['dysfunction'])]
    return relevent

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

##################################    
###### USER INTERFACE ############
##################################

st.title("Générer les resultats de ma campagne")

# input paths

here = os.getcwd()
assets = os.path.join(here, 'assets')
#st.write(f'assets files fetched from: {assets}')
#path to save results
results_folder = os.path.join(here, 'results')
#st.write(f'output files and report saved to: {results_folder}')

single = st.checkbox("Cocher si cet e-diag ne concerne qu'un seule répondant ('Ecart des réponses', 'Cas en marge' et 'points de désaccord' ne seront pas générés).")
if single:
    st.write("Une seule personne a répondu. 'L'écart des réponses', 'Les cas à la marge' et 'Les points de désaccord' non disponibles.")

#st.header("Informations de la campagne")

form_theme = 'Collaborer en Équipe'
api_url = st.secrets["api_url"]
thematics = 'team'

campaign = st.text_input("Entrez le id de la campagne (de type : XX) :")

answers_prefix = st.secrets["answers_prefix"]
answers_url = f'{answers_prefix}/{campaign}'



if campaign:
    st.header("Mes points forts")
    st.write("Voici les dysfonctionnements qui n'ont pas été identifiés")
    atouts = points_forts(answers_url, api_url)
    st.dataframe(data=atouts.head(5))
    st.header("Mes principaux dysfonctionnements")
    top = st.number_input("Entrez le nombre de dysfonctionnements à étudier:")
    topx = int(top)    
    st.write(f"Voici les {topx} principaux dysfonctionnements identifiés :")    
    top5 = dysfunction_frequencies(answers_url, api_url).head(topx)
    top5_dframe = top5.drop(columns=["av_weight"])
    top5_dframe.rename(columns={"dysfunction":"dysfonctionnement"},inplace=True)
    st.dataframe(data=top5_dframe)
    top_5_list = top5_dframe["dysfonctionnement"].values.tolist()

    st.write("Les dysfonctionnements ci-dessous, selon nous, requièrent une attention particulière :")

    most_top5 = most_serious(answers_url, api_url)
    #df[~df['A'].isin([3, 6])]
    most_top5_dframe = most_top5.drop(columns=["weight"])
    most_top5_dframe.rename(columns={"dysfunction":"dysfonctionnement", "people detected":"nombre de personne qui ont identifié ce dysfonctionnement"},inplace=True)
    st.dataframe(data=most_top5_dframe[~most_top5_dframe["dysfonctionnement"].isin(top_5_list)])

    st.header("Vision Globale")

    everyone = augmented_map_dysfunctions(answers_url, api_url)
    if single:
        fig_everyone, ax_everyone = plt.subplots(figsize=(10,4)) 
    else:
        fig_everyone, ax_everyone = plt.subplots(figsize=(10,10))
    sns.heatmap(everyone,annot=False, linewidths=.5, ax=ax_everyone, xticklabels=False, cbar=False, cmap="BuPu")
    plt.tight_layout()

    col1, _ = st.columns([2,1])
    with col1:
        st.pyplot(fig_everyone)

    # save heatmap
    fig_everyone.savefig(f'{results_folder}/campagne_{campaign}_vision_globale.png', dpi=100, transparent=True)
    st.write(f'created: campagne_{campaign}_vision_globale.png ✅')

    vision_globale_page(assets, f"campagne_{campaign}", results_folder)
    with open(f"{results_folder}/campagne_{campaign}_visionglobale.pdf", "rb") as file:
        btn = st.download_button(
            label="Download vision globale page",
            data=file,
            file_name=f"campagne_{campaign}visionglobale.pdf")
    

    st.header('Écart des réponses')
    if single == False:
        team_gap_df = team_gap(answers_url, api_url)
        # create gauge images 
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
        col1, _ = st.columns([2,1])
        with col1:
            st.pyplot(ecart_fig)
    else:
        st.write("Une seule personne a répondu, les cas en marge ne sont pas disponibles")

    st.header("Les cas à la marge")
    if single == False:
        # cas en marge
        cas_marge_df = cas_marge(answers_url, api_url)
        st.write(f"Vous avez {cas_marge_df.shape[0]} cas à la marge. Ci-dessous les {cas_marge_df.shape[0]} dysfonctionnements pour lesquels un répondant n'est pas aligné avec le reste de l'équipe. Cela peut être dû à un isolement du répondant.")
        #show cas marge
        st.dataframe(data=cas_marge_df)
    else:
        st.write("Une seule personne a répondu, les points de désaccord ne sont pas disponibles")

    st.header('Les points de désaccord')
    if single == False:
        # cas en marge
        disagree = augmented_disagree(answers_url, api_url)
        disagree_df = disagree.drop(columns = ["sum", "verdict"])
        st.write(f"Ci-dessous les {disagree_df.shape[0]} dysfonctionnements pour lesquels une moitié de l'équipe n'est pas en accord avec l'autre moitié.")
        # show disagree
        st.dataframe(data=disagree_df)

    st.header("Les solutions proposées")
    solutions = get_all_solutions(assets, answers_url, api_url)
    selection = aggrid_interactive_table(df=solutions)
    filtered_solutions = selection['selected_rows']
    st.write("Les solutions séléctionnées:")
    st.dataframe(data=filtered_solutions)

else:
    st.write("Veuillez renseigner le 'id' de campagne (fourni par Argios) 😉 et appuyez [Enter] ↪️ pour commencer")