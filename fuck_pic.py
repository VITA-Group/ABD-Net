import scipy.io as io
import numpy as np


def pid(s):

    import re
    return re.findall(r'(\d{4}|-1)_', s)[0]


def process(fn):

    dct = io.loadmat(fn)
    distmat = dct['distmat']
    qp = dct['qp']
    gp = dct['gp']

    indices = np.argsort(distmat, axis=1)
    # print(indices)

    errors = []

    for qidx in range(len(qp)):
        # print('index =', qidx)
        # print(qp[qidx])
        qpid = pid(qp[qidx])
        gps = gp[indices[qidx][:5]]
        gpid = [pid(x) for x in gps]
        # if len([x for x in gpid if x == qpid]) < 5:
        #     print('\n'.join(gps))
        errors.append([
            len([x for x in gpid if x == '-1']),
            qidx,
            qp[qidx],
            list(gps),
            gpid
        ])

    return errors


b_error = process('base_distmat.mat')
d_error = process('dan_distmat.mat')
f_error = process('final_distmat.mat')

import os
import shutil
shutil.rmtree('pics', True)
os.makedirs('pics')

lines = []

for be, fe, de in zip(b_error, f_error, d_error):

    qidx = be[1]
    qf = be[2]

    # print(be[3]+fe[3]+de[3])
    if '-1' in be[4] + fe[4] + de[4]:  # be[0] > fe[0] + 2:
        print(qidx, be[0], de[0], fe[0])
        lines.append('{} {} {} {}'.format(qidx, be[0], de[0], fe[0]))
        directory = 'pics/' + str(qidx) + '/'
        os.makedirs(directory)
        shutil.copy(qf.strip(), directory + 'query.jpg')

        for base, paths in [['baseline', be[3]], ['final', fe[3]], ['dan', de[3]]]:
            d = directory + base + '/'
            os.makedirs(d)
            for i, x in enumerate(paths):
                shutil.copy(x.strip(), d + str(i) + '_' + os.path.basename(x.strip()))

with open('pics/error.list', 'w') as f:
    f.write('id baseline_-1_num dan_-1_num final_-1_num\n')
    f.write('\n'.join(lines))
