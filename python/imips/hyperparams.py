# Copyright (C) 2019 Titus Cieslewski, RPG, University of Zurich, Switzerland
#   You can contact the author at <titus at ifi dot uzh dot ch>
# Copyright (C) 2019 Michael Bloesch,
#   Dept. of Computing, Imperial College London, United Kingdom
# Copyright (C) 2019 Davide Scaramuzza, RPG, University of Zurich, Switzerland
#
# This file is part of imips_open.
#
# imips_open is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# imips_open is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with imips_open. If not, see <http:#www.gnu.org/licenses/>.

import absl.flags
import hickle as hkl
import matplotlib.pyplot as plt
import os
import tensorflow as tf

import rpg_datasets_py.dtu
import rpg_datasets_py.euroc
import rpg_datasets_py.hpatches
import rpg_datasets_py.kitti

import baselines
import compress
import datasets
import multiscale
import nets
import sequences
import system


FLAGS = absl.flags.FLAGS


def shortString():
    ret = 'tds=%s_ds=%s' % (FLAGS.tds, FLAGS.ds)
    if FLAGS.rr > 0:
        ret = ret + '_rr=%d' % FLAGS.rr
    if FLAGS.use_qn:
        ret = ret + '_qn'
    if FLAGS.dsup:
        ret = ret + '_dsup'
    if FLAGS.corr_w != 1.:
        ret = ret + '_corrw=%.02f' % FLAGS.corr_w
    if FLAGS.num_scales > 1:
        ret = ret + '_ms=%d,%.2f' % (FLAGS.num_scales, FLAGS.scale_factor)
    if FLAGS.tds == 'hp' and (FLAGS.aug_ang > 0. or FLAGS.aug_sc > 0.):
        ret = ret + '_aug=%.2f-%.2f' % (FLAGS.aug_ang, FLAGS.aug_sc)
    if FLAGS.depth != 12:
        ret = ret + '_d=%d' % FLAGS.depth
    if FLAGS.ol != 0.5:
        ret = ret + '_ol=%.2f' % FLAGS.ol
    if FLAGS.chan != 128:
        ret = ret + '_chan=%d' % FLAGS.chan
    return ret


def bootstrapFromCheckpoint():
    if FLAGS.baseline != '':
        return None, None

    tf.reset_default_graph()
    deploy_net = nets.Deploy2Net()
    saver = tf.train.Saver()

    sess = tf.Session()
    saver.restore(sess, checkpointPath())

    return deploy_net, sess


def checkpointRoot():
    return os.path.join(os.path.dirname(__file__), 'checkpoints')


def checkpointPath():
    base = checkpointRoot()
    if FLAGS.val_best:
        return os.path.join(base, shortString() + '_best')
    else:
        return os.path.join(base, shortString())


def trainStatsPath():
    return checkpointPath() + '_stats'


def getTrainSet():
    if FLAGS.tds == 'hp':
        return rpg_datasets_py.hpatches.HPatches('training', use_min=True)
    elif FLAGS.tds == 'kt':
        assert FLAGS.pck == 'tr'
        ks = [rpg_datasets_py.kitti.KittiSeq(i)
              for i in rpg_datasets_py.kitti.split('training')]
        return sequences.MultiSeqTrackingPairPicker(
            ks, FLAGS.ol, sequences.PairWithIntermediates)
    elif FLAGS.tds == 'rc':
        rs = rpg_datasets_py.robotcar.getSplitSequences('training')
        return sequences.MultiSeqTrackingPairPicker(
            rs, FLAGS.ol, sequences.PairWithIntermediates)
    elif FLAGS.tds == 'tm':
        tms = [rpg_datasets_py.tum_mono.Sequence(i)
               for i in ['01', '02', '03', '48', '49', '50']]
        return sequences.MultiSeqTrackingPairPicker(
            tms, FLAGS.ol, sequences.PairWithIntermediates)
    elif FLAGS.tds == 'tmb':
        tms = [rpg_datasets_py.tum_mono.Sequence('%02d' % i)
               for i in range(1, 51)]
        return sequences.MultiSeqTrackingPairPicker(
            tms, FLAGS.ol, sequences.PairWithIntermediates)
    elif FLAGS.tds == 'dtu':
        dtus = [rpg_datasets_py.dtu.SameLightSequence(i, 4)
                for i in range(1, 25)]
        return datasets.DTUPicker(dtus)
    elif FLAGS.tds == 'en':
        ks = [rpg_datasets_py.kitti.KittiSeq(i)
              for i in rpg_datasets_py.kitti.split('training')]
        rs = rpg_datasets_py.robotcar.getSplitSequences('training')
        tms = [rpg_datasets_py.tum_mono.Sequence(i)
               for i in ['01', '02', '03', '48', '49', '50']]
        return sequences.MultiSeqTrackingPairPicker(
            ks + rs + tms, FLAGS.ol, sequences.PairWithIntermediates)
    else:
        raise NotImplementedError


def getTrainValidationSubset():
    return sequences.FixPairs(getTrainSet(), 50)


def getEvalSequence():
    if evalDs() == 'kt':
        if FLAGS.testing:
            return rpg_datasets_py.kitti.KittiSeq(
                rpg_datasets_py.kitti.split('testing')[0])
        else:
            return rpg_datasets_py.kitti.KittiSeq(
                rpg_datasets_py.kitti.split('validation')[0])
    else:
        assert evalDs() == 'eu'
        assert FLAGS.testing
        return rpg_datasets_py.euroc.EurocSeq('V1_01_easy')


def evalDs():
    return FLAGS.eds if FLAGS.eds is not '' else FLAGS.ds


def label():
    if FLAGS.baseline == '':
        return 'ours(%d)' % FLAGS.chan
    else:
        return methodString()


def getEvalSet(train=False):
    ds = evalDs()
    if not FLAGS.testing:
        if ds == 'hp':
            return rpg_datasets_py.hpatches.HPatches('validation', use_min=True)
        elif ds == 'kt':
            val_seqs = rpg_datasets_py.kitti.split('validation')
            assert len(val_seqs) == 1
            k = rpg_datasets_py.kitti.KittiSeq(val_seqs[0])
            if train:
                pair_class = sequences.PairWithIntermediates
            else:
                pair_class = sequences.PairWithStereo
            rpick = sequences.TrackingPairPicker(
                k, 0.5, pair_class=pair_class)
            return sequences.FixPairs(rpick, 50)
        else:
            raise NotImplementedError
    else:
        assert not train
        print('\n\nTESTING!!!\n\n')
        if ds == 'hp':
            return rpg_datasets_py.hpatches.HPatches('testing', use_min=True)
        elif ds == 'kt':
            val_seqs = rpg_datasets_py.kitti.split('testing')
            assert len(val_seqs) == 1
            k = rpg_datasets_py.kitti.KittiSeq(val_seqs[0])
            rpick = sequences.TrackingPairPicker(
                k, 0.5, pair_class=sequences.PairWithStereo)
            return sequences.FixPairs(rpick, 100)
        elif ds == 'eu':
            seq = rpg_datasets_py.euroc.EurocSeq('V1_01_easy')
            rpick = sequences.TrackingPairPicker(
                seq, 0.5, pair_class=sequences.PairWithStereo)
            return sequences.FixPairs(rpick, 100)
        else:
                raise NotImplementedError


def getForwardPasser(deploy_net=None, sess=None):
    if FLAGS.baseline == '':
        if deploy_net is None:
            deploy_net, sess = bootstrapFromCheckpoint()
        fp = system.ForwardPasser(deploy_net, sess)
        if FLAGS.num_scales == 1:
            return fp
        else:
            return multiscale.ForwardPasser(
                fp, FLAGS.scale_factor, FLAGS.num_scales)
    elif FLAGS.baseline in ['sift', 'surf', 'orb']:
        return baselines.OpenCVForwardPasserMaxN(
            FLAGS.baseline, FLAGS.baseline_num_ips)
    elif FLAGS.baseline == 'sift_pca':
        sift_fp = baselines.OpenCVForwardPasserMaxN(
            'sift', FLAGS.baseline_num_ips)
        m, c, e = hkl.load('compress/tm_sift_pca.hkl')
        pca = compress.PCA(m, c, e)
        return baselines.PCAForwardPasser(sift_fp, pca, FLAGS.pca_dim)
    elif FLAGS.baseline == 'sift_pq':
        sift_fp = baselines.OpenCVForwardPasserMaxN(
            'sift', FLAGS.baseline_num_ips)
        centers = hkl.load('compress/tm_sift_pq_%d_%d_%d.hkl' % (
            FLAGS.pq_m, FLAGS.pq_k, FLAGS.pq_n_init))
        pq = compress.ProductQuantization(centers)
        return baselines.PQForwardPasser(sift_fp, pq)
    else:
        raise NotImplementedError


def methodString():
    if FLAGS.baseline == '':
        return shortString()
    else:
        ret = FLAGS.baseline
        if FLAGS.baseline == 'sift_pq':
            ret = ret + '_%d_%d_%d' % (FLAGS.pq_m, FLAGS.pq_k, FLAGS.pq_n_init)
        if FLAGS.baseline == 'sift_pca':
            ret = ret + '_%d' % FLAGS.pca_dim
        return ret + '_' + '%d' % FLAGS.baseline_num_ips


def evalString():
    ret = evalDs()
    if FLAGS.testing:
        ret = ret + '_testing'
    return ret


def methodEvalString():
    return methodString() + '_' + evalString()


def resultPath(suffix):
    return os.path.join('results', methodEvalString() + suffix)


def resultsOutdated(suffix):
    return os.path.getmtime(checkpointPath() + '.meta') > \
           os.path.getmtime(resultPath(suffix))


def methodNPts():
    if FLAGS.baseline == '':
        return FLAGS.chan
    else:
        return FLAGS.baseline_num_ips


def methodColor():
    cmap = plt.get_cmap('tab10')
    colordict = dict(zip(
        ['', 'sift', 'surf', 'super', 'lfnet', 'sift_pca', 'sift_pq', 'orb'],
        cmap.colors[:8]))
    return colordict[FLAGS.baseline]
