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
    with open(fpath, 'r') as f:
        data = json.load(f)
    return data

def saveJson(dict_:dict, fpath:str):
    with open(fpath, "w") as outfile:
        json.dump(dict_, outfile)


def getTopicModelPath(series_name:str):
    return os.path.join(PATHS['topic_models'], f'topic_model_{series_name}')

def getCustomLabelPath(series_name:str):
    return os.path.join(PATHS['topic_models'], f'custom_labels_{series_name}.json')

def getCharacterTopics(series_name:str, char_name:str):
    return os.path.join(PATHS['topic_models'], f'character_topic_{series_name}_{char_name}.csv')