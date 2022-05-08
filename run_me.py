import os
import random
import numpy as np
import math
from collections import Counter
from timeit import default_timer as timer
from multiprocessing import Pool

nucleotide_to_row = {"A": 0, "C": 1, "G": 2, "T": 3} # global variable

def construct_data_set(ICPC=2, ML=8, SL=500, SC=10, subdirectory="data_set"):
  rand_sequences = []
  for i in range(SC):
    rand_seq = [random.choice("ACGT") for j in range(SL)]
    rand_sequences.append(rand_seq)
  
  mapIPCPtoP = {1.0: 0.8105, 1.5: 0.9245, 2: 1.0}
  pwm = np.zeros(shape=(4, ML))
  for col in range(ML):
    preffered_n = random.randint(0, 3)
    for row in range(4):
      if preffered_n == row:
        pwm[row][col] = mapIPCPtoP[ICPC]
      else:
        pwm[row][col] = (1 - mapIPCPtoP[ICPC]) / 3


  binding_sites = []
  for _ in range(SC):
    binding_site = []
    for col in range(ML):
      p = pwm[:,col]
      binding_site.append(np.random.choice(["A", "C", "G", "T"], p=p))
    binding_sites.append("".join(binding_site))
  
  site_locations = []
  for i in range(SC):
    planting_site = random.randint(0, SL - ML)
    to_plant = random.choice(binding_sites)
    site_locations.append(planting_site)
    for j in range(ML):
      rand_sequences[i][planting_site + j] = to_plant[j]
  
  os.makedirs(subdirectory, exist_ok=True)

  with open(subdirectory+'/sequences.fa', 'w') as f:
    for i in range(len(rand_sequences)):
      f.write(">SEQUENCE_" + str(i+1) + '\n')
      f.write("".join(rand_sequences[i]) + '\n')

  with open(subdirectory+'/sites.txt', 'w') as f:
    for site_location in site_locations:
      f.write(str(site_location) + '\n')
  
  motif = np.matrix(pwm)
  motif = motif.transpose()
  with open(subdirectory+'/motif.txt', 'w') as f:
    f.write(">MOTIF1 " + str(ML) + "\n")
    for line in motif:
        np.savetxt(f, line, fmt='%.4f')
    f.write("<")

  with open(subdirectory+'/motiflength.txt', 'w') as f:
    f.write(str(ML))

################################################################################

def generate_data():
  data_num = 1
  for _ in range(10):
    construct_data_set(subdirectory = "data_set_" + str(data_num))
    data_num += 1
  for ICPC in (1, 1.5, 2):
    for _ in range(10):
      construct_data_set(ICPC=ICPC, subdirectory = "data_set_" + str(data_num))
      data_num += 1
  for ML in (6, 7, 8):
    for _ in range(10):
      construct_data_set(ML=ML, subdirectory = "data_set_" + str(data_num))
      data_num += 1
  for SC in (5, 10, 20):
    for _ in range(10):
      construct_data_set(SC=SC, subdirectory = "data_set_" + str(data_num))
      data_num += 1
  print("DONE!")

################################################################################

def create_PWM(sequences, motif_length, removed_seq_indx):
  pwm = np.zeros(shape=(4, motif_length))

  if removed_seq_indx != -1:
    num_sequences = len(sequences) - 1
  else:
    num_sequences = len(sequences)

  for col in range(motif_length):
    nucleotide_cnt = Counter()
    for i in range(len(sequences)):
      if i == removed_seq_indx:
        continue
      sequence, motif_start = sequences[i]
      nucleotide_cnt[sequence[motif_start + col]] += 1
    for nucleotide in "ACGT":
      pwm[nucleotide_to_row[nucleotide]][col] = nucleotide_cnt[nucleotide] / num_sequences
  return pwm

################################################################################

def compute_information_content(PWM):
  information_content = 0
  for k in range(PWM.shape[1]):
    for beta in range(PWM.shape[0]): # 0=A, 1=C, 2=G, 3=T
      Wbk = PWM[beta][k]
      if Wbk != 0:
        information_content += Wbk * math.log(Wbk / .25, 2)
  return information_content

################################################################################

def sample(sequences, sequence_length, motif_length):
  num_sequences = len(sequences)
  removed_seq_indx = random.randint(0, num_sequences - 1)
  removed_seq = sequences[removed_seq_indx][0]

  pwm = create_PWM(sequences, motif_length, removed_seq_indx)  
  probs = []
  for i in range(sequence_length - motif_length + 1):
    prob = 1
    for j in range(motif_length):
      prob *= pwm[nucleotide_to_row[removed_seq[i + j]]][j]
    probs.append(prob)

  sum_probs = sum(probs)
  if sum_probs == 0:
    normalized_probs = [1 / len(probs)] * len(probs)
  else:
    normalized_probs = [x / sum_probs for x in probs]

  new_motif_start = np.random.choice(range(len(normalized_probs)), p=normalized_probs)
  sequences[removed_seq_indx] = (removed_seq, new_motif_start)
  
  pwm = create_PWM(sequences, motif_length, -1)
  score = compute_information_content(pwm)
  return sequences, pwm, score

################################################################################

def find_motif(subdirectory):
  start = timer()
  with open(subdirectory+'/motiflength.txt', 'r') as f:
    motif_length = int(f.readline())

  sequences = []
  with open(subdirectory+'/sequences.fa', 'r') as f:
    lines = f.readlines()
    sequence_length = len(lines[1].strip())
    for sequence in lines:
      if sequence[0] == ">":
        continue
      motif_start = random.randint(0, sequence_length - motif_length)
      sequences.append((sequence.strip(), motif_start))

  max_score = float('-inf')
  og_sequences = sequences.copy()
  for i in range(100): # ADJUST
    for j in range(10000): # ADJUST
      sequences, pwm, score = sample(sequences, sequence_length, motif_length)
      if score > max_score:
        max_score = score
        best_pwm = pwm
        best_sequences = sequences.copy()
    #print(best_pwm)
    sequences = og_sequences.copy()

  with open(subdirectory+'/predictedsites.txt', 'w') as f:
    for sequence, site_location in best_sequences:
      f.write(str(site_location) + '\n')

  motif = np.matrix(best_pwm)
  motif = motif.transpose()
  with open(subdirectory+'/predictedmotif.txt', 'w') as f:
    f.write(">MOTIF1 " + str(motif_length) + "\n")
    for line in motif:
        np.savetxt(f, line, fmt='%.4f')
    f.write("<")
  end = timer()
  print(subdirectory +" completed in "+str(end - start) +" seconds")
  return 0

################################################################################

if (__name__ == '__main__'):
  generate_data()
  number_of_processes = 8
  list_of_folders = []
  
  #You'll need to run this code twice for optimal results
  #The 1st time will be with the data sets in order
  #The 2nd time will be with the data sets reversed
  #We do this to work around python's multithreading quirks

  #Uncomment the 1st for loop to run the code with data sets in order (1-->100)
  #Uncomment the 2nd for loop to run the code with data sets reversed (100-->1)
  #Don't run the code with both for loops uncommented

  for i in range(1, 101):
    list_of_folders.append("data_set_"+str(i))
  
  #for i in reversed(range(1, 101)):
    #list_of_folders.append("data_set_"+str(i))
  
  #print(list_of_folders)

  with Pool(number_of_processes) as p:
    results = p.map(find_motif, list_of_folders)
    
  print(results)