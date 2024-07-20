import numpy as np
import matplotlib.pyplot as plt
import random
import json
import IPython.display as ipd

import librosa
import librosa.display

import soundfile as sf
import wave

from scipy.io.wavfile import read

import whisper_timestamped as whisper

# bad coding practice but for now assuming sample rate of 22.05 kHz
sr = 22050


def alternate_pitch(y, d_pitch, start_times):
  # alternate doing pitch transformations of d_pitch and keeping original audio
  # d_pitch is the number of semitones to shift the pitch DOWN by
  # e.g. 2 is a 2 semitone decrease in pitch
  shifted_list = []
  shift = False
  i = 0
  # make no changes to audio sample pre-first word
  shifted_list.append(y[:int(start_times[0]*sr)])

  # alternate adjusting pitch by d_pitch and keeping original audio
  while i < len(start_times)-1:
    if shift:
      sample_shifted = librosa.effects.pitch_shift(y = y[int(start_times[i]*sr):int(start_times[i+1]*sr)], sr=sr, n_steps=-d_pitch)
      shift = False
    else:
      sample_shifted = y[int(start_times[i]*sr):int(start_times[i+1]*sr)]
      shift = True
    shifted_list.append(sample_shifted)
    i += 1

  # do the last word 
  if i == len(start_times) -1:
    if shift == True:
      sample_shifted = librosa.effects.pitch_shift(y = y[int(start_times[i]*sr):], sr=sr, n_steps=-d_pitch)
    else:
      sample_shifted = y[int(start_times[i]*sr):]
    shifted_list.append(sample_shifted)

  # create 1D array from list of arrays
  shifted_array = np.concatenate(shifted_list)
  return shifted_array



def all_slow(y, p_slower):
  # slow down whole thing by p_slower %
  return librosa.resample(y, orig_sr=sr, target_sr=sr*(1+p_slower))



def random_word_slow(y, p_slower, start_times):
  # pick one random word and slow it down by p_slower %
  i_slow = random.randint(0, len(start_times)-1)
  i = 0
  shifted_list = []

  while i < len(start_times) -1:
    if i == i_slow:
      sample_shifted = librosa.resample(y[int(start_times[i]*sr):int(start_times[i+1]*sr)], orig_sr=sr, target_sr=sr*(1+p_slower))
      shifted_list.append(sample_shifted)
    else:
      sample_shifted = y[int(start_times[i]*sr):int(start_times[i+1]*sr)]
      shifted_list.append(sample_shifted)
    i += 1

  if i == len(start_times) -1:
    if i == i_slow:
      sample_shifted = librosa.resample(y[int(start_times[i]*sr):], orig_sr=sr, target_sr=sr*(1+p_slower))
    else:
      sample_shifted = y[int(start_times[i]*sr):]
    shifted_list.append(sample_shifted)

  new_sample = np.concatenate(shifted_list)
  return new_sample



def random_word_pitched_down(y, d_pitch, start_times):
  # pick one random word and pitch it down by d_pitch semitones
  i_pitched_down = random.randint(0, len(start_times)-1)
  print(i_pitched_down)
  i = 0
  shifted_list = []

  while i < len(start_times)-1:
    if i == i_pitched_down:
      sample_shifted = librosa.effects.pitch_shift(y = y[int(start_times[i]*sr):int(start_times[i+1]*sr)], sr=sr, n_steps=-d_pitch)
    else:
      sample_shifted = y[int(start_times[i]*sr):int(start_times[i+1]*sr)]
    shifted_list.append(sample_shifted)
    i += 1

  print("out of while")


  if i == len(start_times) -1:
    if i == i_pitched_down:
      sample_shifted = librosa.effects.pitch_shift(y = y[int(start_times[i]*sr):], sr=sr, n_steps=-d_pitch)
    else:
      sample_shifted = y[int(start_times[i]*sr):]
    shifted_list.append(sample_shifted)

  new_sample = np.concatenate(shifted_list)
  return new_sample

def all_slow_random_word_pitched_down(y, p_slower, d_pitch, start_times):
  # slow down whole thing by p_slower % and pitch down one random word by d_pitch semitones
  pitched_down = random_word_pitched_down(y, d_pitch, start_times)
  return all_slow(pitched_down, p_slower)



# POS: https://stackoverflow.com/questions/15388831/what-are-all-possible-pos-tags-of-nltk

def chosen_word_slower(y, p_slower, start_times, words, pos):
  # pick one chosen word and slow it down by p_slower %
  # right now this looks for words with POS tag of adverb, adjective, verb, or noun
  # and shifts only the highest priority one
  # so pos variable in function call does not matter
  # to do: create version where you specify POS - I think this one is good to keep as is though


  import nltk
  nltk.download('averaged_perceptron_tagger')
  words_tagged = nltk.pos_tag(words)
  print(words_tagged)

  words = []
  poss = []
  for pair in words_tagged:
    words.append(pair[0])
    poss.append(pair[1])


  # check for POS in sentence

  if "RB" in poss or "RBR" in poss or "RBS" in poss:
    print("adverb")
    pos = "adverb"
  elif "JJ" in poss or "JJR" in poss or "JJS" in poss:
    print("adjective")
    pos = "adjective"
  elif "VB" in poss or "VBD" in poss or "VBG" in poss or "VBN" in poss or "VBP" in poss or "VBZ" in poss:
    print("verb")
    pos = "verb"
  else:
    print("noun")
    pos = "noun"


  # find the first instance of the POS and save the word
  
  for i in range(len(words_tagged)):
    word = words[i]
    pos_specific = poss[i]
    if pos == "adverb":
      if pos_specific == "RB" or pos_specific == "RBR" or pos_specific == "RBS":
        break
    elif pos == "adjective":
      if pos_specific == "JJ" or pos_specific == "JJR" or pos_specific == "JJS":
        break
    elif pos == "verb":
      if pos_specific == "VB" or pos_specific == "VBD" or pos_specific == "VBG" or pos_specific == "VBN" or pos_specific == "VBP" or pos_specific == "VBZ":
        print("in verb clause")
        break
    else:
      if pos_specific == "NN" or pos_specific == "NNS" or pos_specific == "NNP" or pos_specific == "NNPS":
        break

    print(word)
    print(pos)


  # slow down the word
  i_slow = i
  j = 0
  shifted_list = []

  while j < len(start_times)-1:
    if j == i_slow:
      sample_shifted = librosa.resample(y[int(start_times[j]*sr):int(start_times[j+1]*sr)], orig_sr=sr, target_sr=sr*(1+p_slower))
    else:
      sample_shifted = y[int(start_times[j]*sr):int(start_times[j+1]*sr)]
    shifted_list.append(sample_shifted)
    j += 1


  if j == len(start_times) -1:
    if j == i_slow:
      sample_shifted = librosa.resample(y[int(start_times[j]*sr):], orig_sr=sr, target_sr=sr*(1+p_slower))
    else:
      sample_shifted = y[int(start_times[j]*sr):]
    shifted_list.append(sample_shifted)

  new_sample = np.concatenate(shifted_list)
  return new_sample

def chosen_word_pitched_down(y, d_pitch, start_times, words, pos):
  # POS: https://stackoverflow.com/questions/15388831/what-are-all-possible-pos-tags-of-nltk
  # pick one chosen word and pitch it down by d_pitch semitones
  # see above function for comments on POS tagging
  import nltk
  nltk.download('averaged_perceptron_tagger')
  words_tagged = nltk.pos_tag(words)
  print(words_tagged)

  words = []
  poss = []
  for pair in words_tagged:
    words.append(pair[0])
    poss.append(pair[1])

  if "RB" in poss or "RBR" in poss or "RBS" in poss:
    print("adverb")
    pos = "adverb"
  elif "JJ" in poss or "JJR" in poss or "JJS" in poss:
    print("adjective")
    pos = "adjective"
  elif "VB" in poss or "VBD" in poss or "VBG" in poss or "VBN" in poss or "VBP" in poss or "VBZ" in poss:
    print("verb")
    pos = "verb"
  else:
    print("noun")
    pos = "noun"


  print("pos", pos)

  for i in range(len(words_tagged)):
    word = words[i]
    pos_specific = poss[i]
    if pos == "adverb":
      if pos_specific == "RB" or pos_specific == "RBR" or pos_specific == "RBS":
        break
    elif pos == "adjective":
      if pos_specific == "JJ" or pos_specific == "JJR" or pos_specific == "JJS":
        break
    elif pos == "verb":
      if pos_specific == "VB" or pos_specific == "VBD" or pos_specific == "VBG" or pos_specific == "VBN" or pos_specific == "VBP" or pos_specific == "VBZ":
        print("in verb")
        break
    else:
      if pos_specific == "NN" or pos_specific == "NNS" or pos_specific == "NNP" or pos_specific == "NNPS":
        break


  i_slow = i
  print(i_slow)
  j = 0
  shifted_list = []

  while j < len(start_times)-1:
    if j == i_slow:
      sample_shifted = librosa.effects.pitch_shift(y = y[int(start_times[j]*sr):int(start_times[j+1]*sr)], sr=sr, n_steps=-d_pitch)
    else:
      sample_shifted = y[int(start_times[j]*sr):int(start_times[j+1]*sr)]
    shifted_list.append(sample_shifted)
    j += 1


  if j == len(start_times) -1:
    if j == i_slow:
      sample_shifted = librosa.effects.pitch_shift(y = y[int(start_times[j]*sr):], sr=sr, n_steps=-d_pitch)
    else:
      sample_shifted = y[int(start_times[j]*sr):]
    shifted_list.append(sample_shifted)

  new_sample = np.concatenate(shifted_list)
  return new_sample


def all_slow_chosen_word_pitched_down(y, p_slower, d_pitch, start_times, words, pos):
  # slow down whole thing by p_slower % and pitch down one chosen word by d_pitch semitones
  pitched_down = chosen_word_pitched_down(y, d_pitch, start_times, words, pos)
  return all_slow(pitched_down, p_slower)


def alternate_speed(y, p_slower, start_times):
  # alternate slowing down by p_slower % and keeping original audio
  shifted_list = []
  shift = False
  i = 0
  shifted_list.append(y[:int(start_times[0]*sr)])

  while i < len(start_times)-1:
    print(i)
    if shift:
      sample_shifted = librosa.resample(y[int(start_times[i]*sr):int(start_times[i+1]*sr)], orig_sr=sr, target_sr=sr*(1+p_slower))
      shift = False
    else:
      sample_shifted = y[int(start_times[i]*sr):int(start_times[i+1]*sr)]
      shift = True
    shifted_list.append(sample_shifted)
    i += 1

  if i == len(start_times):
    if shift == True:
      sample_shifted = librosa.resample(y[int(start_times[i]*sr):int(start_times[i+1]*sr)], orig_sr=sr, target_sr=sr*(1+p_slower))
    else:
      sample_shifted = y[int(start_times[j]*sr):int(start_times[i+1]*sr)]
    shifted_list.append(sample_shifted)

  print(len(shifted_list))
  shifted_array = np.concatenate(shifted_list)

  return shifted_array


def all_pitched_down(y, d_pitch):
  # pitch down whole thing by d_pitch semitones
  return librosa.effects.pitch_shift(y = y, sr=sr, n_steps=-d_pitch)


def all_pitched_down_random_slowed(y, d_pitch, p_slower, start_times):
  # pitch down whole thing by d_pitch semitones and slow down one random word by p_slower %
  random_word_slowed = random_word_slow(y, p_slower, start_times)
  return all_pitched_down(random_word_slowed, d_pitch)


def all_pitched_down_chosen_word_slowed(y, d_pitch, p_slower, start_times, words, pos):
  # pitch down whole thing by d_pitch semitones and slow down one chosen word by p_slower %
  chosen_word_slowed = chosen_word_slower(y, p_slower, start_times, words, pos)
  return all_pitched_down(chosen_word_slowed, d_pitch)