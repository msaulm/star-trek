import os
import json

# paths to relevant folders and files
cwd = os.getcwd()
assert(cwd.split('/')[-1]=='star-trek'), "incorrect root folder"
PATHS = {}
PATHS['data'] = os.path.join(cwd,'data')
PATHS['series_lines'] = os.path.join(PATHS['data'],'all_series_lines.json')
PATHS['topic_models'] = os.path.join(PATHS['data'],'topic_models')

# common functions
def loadJson(fpath:str)->dict:
    """load json given path

    Args:
        fpath (str): path to json

    Returns:
        dict: loaded json
    """    
    # load json file    
    with open(fpath, 'r') as f:
        data = json.load(f)
    return data

def saveJson(dict_:dict, fpath:str):
    """save dict as json to fpath

    Args:
        dict_ (dict): dict to save
        fpath (str): path to save to
    """    
    #save json file
    with open(fpath, "w") as outfile:
        json.dump(dict_, outfile)


def getTopicModelPath(series_name:str)->str:
    """parse series name to derive path to respective topic model 

    Args:
        series_name (str): name of series

    Returns:
        str: path to topic model
    """    
    return os.path.join(PATHS['topic_models'], f'topic_model_{series_name}')

def getCustomLabelPath(series_name:str)->str:
    """parse series name to derive path to respective custom label json 

    Args:
        series_name (str): name of series

    Returns:
        str: path to custom label json
    """    
    return os.path.join(PATHS['topic_models'], f'custom_labels_{series_name}.json')

def getCharacterTopics(series_name:str, char_name:str)->str:
    """parse series and char name to derive path to respective character topics path 

    Args:
        series_name (str): name of series
        char_name (str): name of character

    Returns:
        str: path to character topics path
    """    
    return os.path.join(PATHS['topic_models'], f'character_topic_{series_name}_{char_name}.csv')

def getSeriesTopics(series_name:str)->str:
    """parse series name to derive path to series topics 

    Args:
        series_name (str): name of series

    Returns:
        str: path to series topics
    """    
    return os.path.join(PATHS['topic_models'], f'series_topic_{series_name}.csv')