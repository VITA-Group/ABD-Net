from MulticoreTSNE import MulticoreTSNE as TSNE
import scipy.io as io
import numpy as np
from numpy.linalg import norm

dct = {}

for name, fn in [['base', 'base_feat.mat'], ['dan', 'dan_feat.mat'], ['final', 'final_feat.mat']]:

    mat = io.loadmat(fn)
    t = mat['gt']
    tsne = TSNE(n_jobs=32)

    features = mat['g']
    # features = features / (norm(features, axis=1, keepdims=True))

    a = tsne.fit_transform(mat['g'])
    dct.update({
        name + '_a': a,
        name + '_t': t,
    })

io.savemat('tsne.mat', dct)
