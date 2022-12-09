#%%
from bertopic import BERTopic
from helper_functions import PATHS
from helper_functions import loadJson
from helper_functions import saveJson
from helper_functions import getTopicModelPath
from helper_functions import getCustomLabelPath
from helper_functions import getCharacterTopics
import pandas as pd
import flatdict
import os
import numpy as np
#%%
# load data
'''
    Star Trek The Original Series (TOS) (1966)
    Star Trek The Animated Series (TAS) (1973)
    Star Trek The Next Generation (TNG) (1987)
    Star Trek Deep Space Nine (DS9)     (1993)    
    Star Trek Voyager (VOY)             (1995)
    Star Trek Enterprise (ENT)          (2001)
allseriesline={seriesname:{episode number:{character:alllines}}}
'''
series_order = ['TOS','TAS','TNG','DS9','VOY','ENT']
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
                                                                'episode_count': 0,
                                                                'lines': [],}
            n_cw = sum([len(line.split(' ')) for line in character_lines if len(line)>0])
            n_cl = len([line for line in character_lines if len(line)>0])
            series_character_lines[series][character]['line_count']+=n_cl
            series_character_lines[series][character]['word_count']+=n_cw
            series_character_lines[series][character]['episode_count']+=1

            for line in character_lines:
                if len(line)>0:
                    series_character_lines[series][character]['lines'].extend([line])

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
    # main characters are assumed to appear the most and have the most to say
    most_episode_and_word_count = series_character_stats_df.sort_values(['episode_count','word_count'], ascending=False).iloc[0:5]
    series_character_stats_dict[series] = { 'top_5_by_lines': list(top_5_by_lines.index),
                                            'most_verbose_by_word_count': most_verbose_by_word_count.name,
                                            'most_verbose_by_line_normalized_word_count': most_verbose_by_line_normalized_word_count.name,
                                            'most_episode_and_word_count': list(most_episode_and_word_count.index)}

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
        custom_labels = {int(k): v for k,v in custom_labels.items()}
        custom_label_collection[series] = custom_labels
    else:
        print(f'processing custom label {series}')
        print(topic_models[series].get_topic_info())
        custom_labels = {}
        for i in range(0,10):
            print(topic_models[series].get_topic(i))
            name = input()
            custom_labels[i] = name
        custom_label_collection[series] = custom_labels
        saveJson(custom_labels, custom_label_path)

#%%
# who are the main characters
# the main characters should be the ones that are in most episodes and have highest word counts
main_character_main_topics = {}
for series, character_stats in series_character_stats_dict.items():
    main_characters = character_stats['most_episode_and_word_count']
    main_character_main_topics[series] = {}
    for character in main_characters:
        character_topics_df_path = getCharacterTopics(series, character)
        if os.path.exists(character_topics_df_path):
            character_topics_df = pd.read_csv(character_topics_df_path)
        else:        
            character_topic_prediction = topic_models[series].transform(series_character_lines[series][character]['lines'])
            character_topics_df = pd.DataFrame(character_topic_prediction).T.rename(columns={0:'topic',1:'p'})
            character_topics_df['topic'] = character_topics_df['topic'].astype(int)
            character_topics_df.to_csv(character_topics_df_path, index=False)
        topic_counts = character_topics_df.query("p>0.7").value_counts('topic').reset_index().rename(columns={0:'count'}) # lines that have an assigned topic probability of >0.5 are considered
        
        topic_count_main = np.quantile(topic_counts.query('topic>=0')['count'],0.99) # main topics are in the top 10th percentile of discussed topic by character
        main_discussed_topics = topic_counts.query('topic>=0').query(f"count>={topic_count_main}")

        main_character_main_topics[series][character] = []
        for topic in main_discussed_topics['topic']:
            if topic in custom_label_collection[series].keys():
                topic_label = custom_label_collection[series][topic]
            else:
                topic_label = topic_models[series].topic_labels_[topic]
            main_character_main_topics[series][character].append(topic_label)

#%%
# topics through out series
series_topics = {}
for series in series_order:
    series_topics[series] = []
    topic_info_for_series = topic_models[series].get_topic_info().query('Topic>=0')
    main_series_topics = topic_info_for_series.iloc[0:10]
    for topic in main_series_topics['Topic']:
        if topic in custom_label_collection[series].keys():
            topic_label = custom_label_collection[series][topic]
        else:
            topic_label = topic_models[series].topic_labels_[topic]
        series_topics[series].append(topic_label)

unique_topics = np.unique([topic for s,s_v in series_topics.items() for topic in s_v ])
pd.DataFrame(columns=[series_order], index=unique_topics)
print('pause')
# for series, series_characters_lines in series_character_lines.items():
#     for character, character_stats in series_characters_lines.items():
#         topic_prediction = topic_models[series].transform(character_stats['lines'])
#         series_character_lines[series][character].update({'topics': topic_prediction})