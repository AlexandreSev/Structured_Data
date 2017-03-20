#coding : utf-8


"""
Let's assume that we have a function that compute the score for the model1, let's say predict1, and
one for the model 2, predict2. We want to implement a beam search to find a good result. 
"""    

list_character = "abcdefghijklmnopqrstuvwxyz0123456789_"

def predict1(picture):
    return np.ones((23, 30))

def predict2(picture):
    return np.ones(10000)

dico_grams = {'a': 0, 'b':1}
dico_grams_reverse = {0: 'a', 1:'b'}

def next_step(list_candidates, prediction1, prediction2, threegrams=True):
    new_candidates = []
    for score, charac in list_candidates:
        for indic in range(37):
            score_temp = prediction1[indic]
            for char in list_character:
                temp = charac[-1] + char
                indic_temp = dico_grams[temp]
                score_temp_temp = score_temp + prediction2[indic_temp]
                if threegrams:
                    temp = charac[-2:] + char
                    indic_temp = dico_grams[temp]
                    score_temp_temp += prediction2[indic_temp]
                new_candidates.append((score_temp_temp, charac + char))
                new_candidates.sort(reverse=True)
                new_candidates = new_candidates[0:10]
    return new_candidates


def beam_search(picture, beam_size=10):
    prediction1 = predict1(picture)
    prediction2 = predict2(picture)
    
    list_score = [prediction1[i] + prediction2[dico_grams[i]] for i in range(37)]
    list_character = "abcdefghijklmnopqrstuvwxyz0123456789_"
    beam_list = list_candidate[0: 10]
    list_candidates = zip(list_score, list_character)
    list_candidates.sort(reverse=True)
    list_candidates = list_candidates[0:10]

    list_candidates = next_step(list_candidates, prediction1, prediction2, False)

    for i in range(21):
        list_candidates = next_step(list_candidates, prediction1, prediction2)

    return list_candidates


