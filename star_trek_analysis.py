#%%
from bertopic import BERTopic
from helper_functions import PATHS
from helper_functions import loadJson
from helper_functions import saveJson
from helper_functions import getTopicModelPath
from helper_functions import getCustomLabelPath
import pandas as pd
import flatdict
import os
import numpy as np
import json
#%%
# load data
'''
    Star Trek The Original Series (TOS)
    Star Trek The Animated Series (TAM)
    Star Trek The Next Generation (TNG)
    Star Trek Deep Space Nine (DS9)
    Star Trek Voyager (VOY)
    Star Trek Enterprise (ENT)
allseriesline={seriesname:{episode number:{character:alllines}}}
'''
series_lines_dict = loadJson(PATHS['series_lines'])
#%%
# There are different series, the assumption is that there are different 
# characters in each series, so it would make sense to analyse characters on
# a series by series basis
series_character_lines = {}
for series, series_dict in series_lines_dict.items():
    print(f"Series: {series}")
    series_character_lines[series] = {}
    for episode, episode_dict in series_dict.items():
        for character, character_lines in episode_dict.items():
            if character not in series_character_lines[series].keys():
                series_character_lines[series][character] = {   'line_count': 0,
                                                                'word_count': 0,
                                                                'lines': []}
            n_cw = sum([len(line.split(' ')) for line in character_lines])
            n_cl = len(character_lines)
            series_character_lines[series][character]['line_count']+=n_cl
            series_character_lines[series][character]['word_count']+=n_cw
            joined_lines = ' '.join(character_lines)
            if len(joined_lines)>0:
                series_character_lines[series][character]['lines'].append(' '.join(character_lines))

# for each series, who has most lines (top 5), is the most verbose
series_character_stats_dict = {}
for series, series_character_stats in series_character_lines.items():
    series_character_stats_df = pd.DataFrame(series_character_stats).T

    consider_index = series_character_stats_df.query('line_count>0').index
    series_character_stats_df['word_per_line'] = np.nan
    series_character_stats_df.loc[consider_index,'word_per_line'] = series_character_stats_df.loc[consider_index, 'word_count']/series_character_stats_df.loc[consider_index, 'line_count']

    top_5_by_lines = series_character_stats_df.sort_values('line_count', ascending=False).iloc[0:5]
    most_verbose_by_word_count = series_character_stats_df.sort_values('word_count', ascending=False).iloc[0]
    most_verbose_by_line_normalized_word_count = series_character_stats_df.sort_values('word_per_line', ascending=False).iloc[0]

    series_character_stats_dict[series] = { 'top_5_by_lines': list(top_5_by_lines.index),
                                            'most_verbose_by_word_count': most_verbose_by_word_count.name,
                                            'most_verbose_by_line_normalized_word_count': most_verbose_by_line_normalized_word_count.name}

#%%
# the assumption is that each series takles different themes and topics
def groupTextForDimension(data_dict, dimension, dimension_value):
    if dimension=='chracter':
        pass
    if dimension=='episode':
        flattened_series = flatdict.FlatDict(series_lines_dict[dimension_value])
    if dimension=='series':
        flattened_series = flatdict.FlatDict(data_dict[dimension_value])
        corpus = [text for key, text in flattened_series.items()]
    return corpus

topic_models = {}
for series, series_dict in series_lines_dict.items():
    series_corpus = []
    for episode, episode_v in series_dict.items():
        for _, character_line in episode_v.items():
            if len(character_line)>0:
                series_corpus.extend(character_line)

    topic_model_path = getTopicModelPath(series)
    print(f"getting topic model for series {series}")
    if os.path.exists(topic_model_path):
        topic_model = BERTopic.load(topic_model_path)
    else:
        topic_model = BERTopic()
        topic_model.fit_transform(series_corpus)
        topic_model.save(topic_model_path)
    topic_models[series] = topic_model

# set custom labels
custom_label_collection = {}
for series in topic_models.keys():
    custom_label_path = getCustomLabelPath(series)
    if os.path.exists(custom_label_path):
        print(f'loading custom label {series}')
        custom_labels = loadJson(custom_label_path)
    else:
        print(f'processing custom label {series}')
        print(topic_models[series].get_topic_info())
        custom_labels = {}
        for i in range(0,10):
            print(topic_models[series].get_topic(i))
            name = input()
            custom_labels[i] = name
        custom_label_collection[series] = custom_labels
        print(custom_labels)
        saveJson(custom_labels, custom_label_path)
    topic_models[series].set_topic_labels(custom_labels)

# who are the main characters

for series, series_characters_lines in series_character_lines.items():
    for character, character_stats in series_characters_lines.items():
        topic_prediction = topic_models[series].transform(character_stats['lines'])
        series_character_lines[series][character].update({'topics': topic_prediction})