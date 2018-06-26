import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from tqdm import tqdm
import sys
import time
import readline


ENC = "https://tfhub.dev/google/universal-sentence-encoder-large/1"
BATCH = 4000

# -------------------------------------------------------------------------------

def rel_dist( pts, x, pts2 ):
    return pts2 - 2.0*np.dot( pts, x ) + np.dot( x.T, x )

# -------------------------------------------------------------------------------

class file_encoder():
    def __init__(self, fn):

        with tf.device('/cpu:0'):
            dataset = tf.data.TextLineDataset( fn )
            dataset = dataset.batch( BATCH )
            dataset = dataset.prefetch( 2 )
            iterator = dataset.make_one_shot_iterator()
            self.strings = iterator.get_next()

        with tf.device('/gpu:0'):
            self.embed = hub.Module( ENC )
            self.embed_ds = self.embed( self.strings )

        self.sess = tf.Session( config=tf.ConfigProto(allow_soft_placement=True) )
        self.sess.run( tf.global_variables_initializer() )
        self.sess.run( tf.tables_initializer() )
        
    def embed_file( self, maxcnt=np.Inf ):
        embed_data = []
        line_data = []
    
        pbar = tqdm( total=maxcnt )
        cnt = 0
        while True:

            r, lines = self.sess.run( [ self.embed_ds, self.strings ] )
            
            line_data.append( lines )
            embed_data.append( r )

            cnt += len( lines )
            pbar.update( len(lines) )

            if cnt > maxcnt or len(lines) < BATCH:
                break

        flat_list = [item for sublist in line_data for item in sublist]
        embed_data = np.vstack( embed_data )
    
        return flat_list, embed_data

# -------------------------------------------------------------------------------

class encoder:
    def __init__(self):

        with tf.device('/gpu:0'):
            self.embed = hub.Module( ENC )
            self.sen_ph = tf.placeholder( tf.string, shape=(None) )
            self.embed_one = self.embed( self.sen_ph )

        self.sess = tf.Session( config=tf.ConfigProto(allow_soft_placement=True) )
        self.sess.run( tf.global_variables_initializer() )
        self.sess.run( tf.tables_initializer() )

    def em( self, sentence ):
        return np.atleast_2d( self.sess.run( self.embed_one, feed_dict={ self.sen_ph: [sentence] } ) )

    def ems( self, sentences ):
        return self.sess.run( self.embed_one, feed_dict={ self.sen_ph: sentences } )

# -------------------------------------------------------------------------------

class dataset:
    def __init__(self, lines, pts, encoder):
        self.lines = lines
        self.pts = pts
        self.pts2 = np.sum( pts * pts, axis=1, keepdims=True )
        self.encoder = encoder

    def nearest_sen( self, sentence ):
        sen_em = self.encoder.em( sentence )

        return nearest_emb( self, sen_em )

    def nearest_emb( self, sen_em ):
        dists = rel_dist( self.pts, sen_em.T, self.pts2 )
        closest_ind = np.argmin( dists )
        return self.lines[ closest_ind ]
    
# -------------------------------------------------------------------------------

def test_em( ds ):
    ind = np.random.randint( len(lines) )
    sen = lines[ ind ]
    print( "Orig: " + sen )
    closest_sen = nearest( sen, lines, pts, pts2 )
    print( "New: " + closest_sen )

def an( s1, s2, s3, ds ):
    es1 = ds.em( s1 )
    es2 = ds.em( s2 )
    es3 = ds.em( s3 )

    v = es1 - es2 + es3

    return ds.nearest_emb( v )

# -------------------------------------------------------------------------------

def simple_chat( ds ):
    debug = False
    CNT = 5
    while True:
        line = raw_input( "> " ).rstrip()
        if line == "":
            continue
        if line == "debug on":
            debug = True
            continue
        if line == "debug off":
            debug = False
            continue
        if line == "quit":
            break

        t_start = time.time()
        sen_em = ds.encoder.em( line )
        t_mid = time.time()
        
        dists = rel_dist( ds.pts, sen_em.T, ds.pts2 )
        inds = np.argsort( dists.ravel() )

        t_end = time.time()
        
        if not debug:
            response_ind = np.random.randint( CNT )
            sys.stdout.write( ds.lines[ inds[response_ind]+1] + "\n" )
        else:
            for ind in range(CNT):
                sys.stdout.write( "  nearest embed: [" + ds.lines[ inds[ind] ] + "]\n" )
                sys.stdout.write( "       response: [" + ds.lines[ inds[ind]+1] + "]\n" )
                sys.stdout.write( "       distance: [ %.2f ]\n" % dists[inds[ind]] )
            sys.stdout.write( "   elapsed time: [ %.4f + %.4f = %.4f ]\n" % (t_mid-t_start,t_end-t_mid,t_end-t_start) )

def simple_chat2( ds ):
    debug = False
    CNT = 5
    while True:
        line = raw_input( "> " ).rstrip()
        if line == "":
            continue
        if line == "debug on":
            debug = True
            continue
        if line == "debug off":
            debug = False
            continue
        if line == "quit":
            break

        # find the nearest sentences to the input
        t_start = time.time()
        sen_em = ds.encoder.em( line )
        t_mid = time.time()
        dists = rel_dist( ds.pts, sen_em.T, ds.pts2 )
        inds = np.argsort( dists.ravel() )
        t_end = time.time()

        # average the responses, and recompute
        sen_em = np.mean( ds.pts[ inds[0:CNT]+1, : ], axis=0, keepdims=True )
        dists = rel_dist( ds.pts, sen_em.T, ds.pts2 )
        inds = np.argsort( dists.ravel() )
        
        if not debug:
            response_ind = np.random.randint( CNT )
            sys.stdout.write( ds.lines[ inds[response_ind] ] + "\n" )
        else:
            for ind in range(CNT):
                sys.stdout.write( "  nearest embed: [" + ds.lines[ inds[ind] ] + "] [%.2f]\n" % dists[inds[ind]] )
            sys.stdout.write( "   elapsed time: [ %.4f + %.4f = %.4f ]\n" % (t_mid-t_start,t_end-t_mid,t_end-t_start) )

#enc = file_encoder( "/mnt/pccfs/not_backed_up/data/bookcorpus/books_large_p1.txt" )
#lines, pts = enc.embed_file( 10000000 )

#lines = np.load( "/mnt/pccfs/not_backed_up/data/chitchat/processed/ccc_lines.npy" )
#pts = np.load( "/mnt/pccfs/not_backed_up/data/chitchat/processed/ccc_pts.npy" )

#fenc = file_encoder( "all_ccc_super_short.txt" )
#lines, pts = fenc.embed_file( 10000000 )

lines = np.load( "./ccc_ss_lines.npy" )
pts = np.load( "./ccc_ss_pts.npy" )

enc = encoder()
ds = dataset( lines, pts, enc )
#simple_chat( ds )
simple_chat2( ds )