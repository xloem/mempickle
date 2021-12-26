

import retrain

reer = retrain.Retrainer('/mnt/4/models/gpt2-ptmap', '/mnt/4/models/gpt2-ptmap16')
mods = reer.mods_by_difference(shallowness=2)
