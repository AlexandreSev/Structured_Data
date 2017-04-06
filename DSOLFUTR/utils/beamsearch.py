#coding : utf-
from .utils import get_score

def next_step(list_candidates, prediction1, prediction2, list_character, dico_ngrams, dico_conversion1, 
                n_iter, threegrams=True, beam_size=10):
    """
    Compute one step of the beam_search.
    Parameters:
        list_candidates: list of tuple. The first element of each tuple is the current score, the second
                            one is the current string
        prediction1: prediction from the model 1
        prediction2: prediction from the model 2
        list_character: a list of all possible characters
        dico_ngrams: dico matching a n-gram to its position in model 2 output
        dico_conversion1: dico matching a character to its position in model 1 output
        n_iter: number of the current iteration
        threegrams: Boolean. IF true, will search on three-grams and on two-grams
        beam_size. Size of the queue to consider
    """
    new_candidates = []
    for score, charac in list_candidates:   
        for char in list_character:
            score_temp = prediction1[n_iter, dico_conversion1[char]] + score 
            score_temp += get_score(prediction2, dico_ngrams, char)
            temp = charac[-1] + char
            score_temp_temp = score_temp + get_score(prediction2, dico_ngrams, temp)
            if threegrams:
                temp = charac[-2:] + char
                score_temp_temp += get_score(prediction2, dico_ngrams, temp)
            new_candidates.append((score_temp_temp, charac + char))
        new_candidates.sort(reverse=True)
        new_candidates = new_candidates[0:beam_size]
    return new_candidates


def beam_search(prediction1, prediction2, list_character, dico_ngrams, dico_conversion1, beam_size=10):
    """
    Compute the full beamsearch
    Parameters:
        prediction1 : prediction provided by model1
        preidction2 : prediction provided by model2
        dico_ngrams : dictionnary matching a ngram to its position in prediction2
        dico_conversion : dictionnary matching characters to their position in prediction1
        beam_size : the size of the list kept in memory
    """

    list_score = [prediction1[0, dico_conversion1[i]] + get_score(prediction2, dico_ngrams, i) for i in list_character]
    
    list_candidates = list(zip(list_score, list_character))
    list_candidates.sort(reverse=True)
    list_candidates = list_candidates[0:beam_size]

    list_candidates = next_step(list_candidates, prediction1, prediction2, list_character, dico_ngrams, 
                                dico_conversion1, 1, False, beam_size)

    for i in range(21):
        list_candidates = next_step(list_candidates, prediction1, prediction2, list_character, 
                                    dico_ngrams, dico_conversion1, 2 + i, True, beam_size)

    return list_candidates[0][1]

def build_prediction(prediction):
    return str("".join(prediction))
