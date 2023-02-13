# lumen-audio

## Terminologija

Pristup spektogram CNN:

- uzmi uzorak
- rascjepaj ga na djelove po X sekundi
- za svaki napravi spektogram
- CNN inference
  Uzorak (3sec)
  Sekvenca, sastoji se od više uzoraka
  Dulji uzorak, validacijski skup (5-20sec)

how to create a sequence?

- sakupi sve instrumente
- napravi jednu sekvencu tako da uzmeš nekoliko .wavova
-

uniform sample?

sequence -> Model -> classify

istreniraj model prvo na čistim instrumentima

finetunaj na sekvencama

Kako enkodidarti žanr? Napraviti array klasa
array \[dru, nod | cou-fol, cla \] has to be sent in the model

## Dataset

Test set (ours) contains only information about presence of the instruments, genre and drums are not present.

## Preprocess

to one channel
input to model: Mel-spectogram
Normazlie spectograms (mean/std)

## Augmentation

white nose before going to spectral dimension
