#coding : utf-
from .utils import get_score

def next_step(list_candidates, prediction1, prediction2, list_character, dico_ngrams, 
              threegrams=True, beam_size=10):
    new_candidates = []
    for score, charac in list_candidates:
        for indic in range(37):
            score_temp = prediction1[indic]
            for char in list_character:
                temp = charac[-1] + char
                indic_temp = dico_ngrams.get(temp, 0)
                score_temp_temp = score_temp + prediction2[indic_temp]
                if threegrams:
                    temp = charac[-2:] + char
                    indic_temp = dico_ngrams.get(temp, 0)
                    score_temp_temp += prediction2[indic_temp]
                new_candidates.append((score_temp_temp, charac + char))
                new_candidates.sort(reverse=True)
                new_candidates = new_candidates[0:beam_size]
    return new_candidates


def beam_search(prediction1, prediction2, list_character, dico_ngrams, dico_conversion1, beam_size=10):

    list_score = [prediction1[dico_conversion1[i]] + get_score(prediction2, dico_ngrams, i) for i in list_character]
    list_character = "abcdefghijklmnopqrstuvwxyz0123456789_"
    
    list_candidates = list(zip(list_score, list_character))
    list_candidates.sort(reverse=True)
    list_candidates = list_candidates[0:beam_size]

    list_candidates = next_step(list_candidates, prediction1, prediction2, list_character, dico_ngrams, 
                                False, beam_size)

    for i in range(21):
        list_candidates = next_step(list_candidates, prediction1, prediction2, list_character, dico_ngrams, 
                                True, beam_size)

    return list_candidates[0][1]


