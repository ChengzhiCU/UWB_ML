import numpy as np


def filter_d(wave, feature, label):
  rise_time = feature[:,5]
  maxamp = feature[:,6]
  mask1 = rise_time < 0
  mask2 = rise_time > 20
  mask = mask1 + mask2
  maskb = ~mask
  rise_timeb = rise_time[maskb]
  featureb = feature[maskb]
  waveb = wave[maskb]
  labelb = label[maskb]
  maxamp = featureb[:,6]
  maskmax = maxamp > 500
  rise_timec = rise_timeb[maskmax]
  featurec = featureb[maskmax]
  wavec = waveb[maskmax]
  labelc = labelb[maskmax]
  maxamp2 = featurec[:,6]
  maskmax2 = maxamp2 < 10000
  rise_timed = rise_timec[maskmax2]
  featured = featurec[maskmax2]
  waved = wavec[maskmax2]
  labeld = labelc[maskmax2]
  Er = featured[:,1]
  Ermin = Er >1000
  featuree = featured[Ermin]
  wavee = waved[Ermin]
  labele = labeld[Ermin]


  return wavee,featuree,labele