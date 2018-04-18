from config import *
import numpy as np
import os

lab = np.load(os.path.join(PAESED_FILES_6F_NLOS, 'all_436.npy'))[()]

extracted_features_lab = lab['extracted_features']
wave_lab = lab['wave']
print('labeled', np.mean(extracted_features_lab, axis=0), np.std(extracted_features_lab, axis=0))


lab = np.load(os.path.join(LOS_PAESED_FILES_NEW, 'all_258.npy'))[()]

extracted_features_lab = lab['extracted_features']
wave_lab = lab['wave']
print('los new labeled', np.mean(extracted_features_lab, axis=0), np.std(extracted_features_lab, axis=0))


unl = np.load(os.path.join(UNLABELED_PARSED, 'unlabeled_11.npy'))[()]
extracted_features_unl = unl['extracted_features']
wave_unl = unl['wave']
print('unlabeled', np.mean(extracted_features_unl, axis=0), np.std(extracted_features_unl, axis=0))

