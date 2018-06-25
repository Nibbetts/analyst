import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from tqdm import tqdm
import sys
import scipy as sp
import re

embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/1")
sess = tf.Session()
sess.run( tf.global_variables_initializer() )
sess.run( tf.tables_initializer() )
sen_ph = tf.placeholder( tf.string, shape=(None) )
embeddings = embed( sen_ph )

# -------------------------------------------------------------------------------

def em( sentence ):
    return sess.run( embeddings, feed_dict={ sen_ph: [sentence] } )

# -------------------------------------------------------------------------------

def test_em( lines, pts ):
    ind = np.random.randint( len(lines) )

    sen = lines[ ind ]
    print( "Orig: " + sen )
    closest_sen = nearest( sen, lines, pts )
    print( "New: " + closest_sen )
    
def nearest( sentence, lines, pts ):
    sen_em = em( sentence )
    sen_em = np.atleast_2d( sen_em )
    
    dists = sp.spatial.distance.cdist( pts, sen_em, metric='euclidean' )
    closest_ind = np.argmin( dists )
    return lines[ closest_ind ]

def an( s1, s2, s3, lines, pts ):
    es1 = em( s1 )
    es2 = em( s2 )
    es3 = em( s3 )

    v = es1 - es2 + es3

    dists = sp.spatial.distance.cdist( pts, v, metric='euclidean' )
    closest_ind = np.argmin( dists )
    return lines[ closest_ind ]

def embed_file( fn ):
    datafile = open( fn, 'rb' )

    embed_data = []
    line_data = []
    
    pbar = tqdm()
    ind = 0
    while True:
    
        lines = [ x.rstrip() for x in datafile.readlines( 32000 ) ]
        line_cnt = len( lines )
        if line_cnt == 0:
            break
        
        line_data.append( lines )
    
        r = sess.run( embeddings, feed_dict={ sen_ph: lines } )
    
        embed_data.append( r )

        ind += line_cnt
        pbar.update( line_cnt )

    flat_list = [item for sublist in line_data for item in sublist]

    embed_data = np.vstack( embed_data )
    
    return flat_list, embed_data

def simple_chat( lines, pts, answer_lines, answer_pts, variation=5 ):
    debug = True
    CNT = variation
    while True:
        sys.stdout.write( "> " )
        line = sys.stdin.readline().rstrip()
        if line == "":
            continue
        if line == "debug on":
            debug = True
            continue
        if line == "debug off":
            debug = False
            continue
        if line == "quit" or line == "q":
            break

        sen_em = em( line )
        #sen_em = np.atleast_2d( sen_em )
    
        #dists = sp.spatial.distance.cdist( pts, sen_em, metric='euclidean' )
        #inds = np.argsort( dists.ravel() )
        # inds = np.argpartition( dists.ravel(), range( CNT ) )

        # if not debug:
        #     response_ind = np.random.randint( CNT )
        #     sys.stdout.write( lines[ inds[response_ind]+1] + "\n" )
        # else:
        #     for ind in range(CNT):
        #         sys.stdout.write( "  nearest embed: [" + lines[ inds[ind] ] + "]\n" )
        #         sys.stdout.write( "       response: [" + lines[ inds[ind]+1] + "]\n" )
        #         sys.stdout.write( "       distance: [ %.2f ]\n" % dists[inds[ind]] )

        dists = sp.spatial.distance.cdist( pts, np.atleast_2d( sen_em ), metric='euclidean' )
        dists = dists.ravel()
        pre_inds = np.argpartition( dists, range( CNT ) )[:CNT]
        pre_inds = pre_inds[dists[pre_inds] < 0.68]
        if len(pre_inds) > 0:
            pre_inds_follow = pre_inds + 1
            #pre_inds_hist = pre_inds - 1

            ##nearest_ind = np.argmin(dists.ravel())
            ##nearest_vec = pts[ nearest_ind ]
            ##followup_vec = pts[ nearest_ind + 1 ]
            #start_vec = np.mean(pts[ pre_inds ], axis=0)
            #follow_vec = np.mean(pts[ pre_inds_follow ], axis=0)
            #hist_vec = np.mean(pts[ pre_inds_hist ], axis=0)
            analogy_vecs = pts[ pre_inds_follow ] - pts[ pre_inds ]
            #hist_analogy_vecs = pts[ pre_inds ] - pts[ pre_inds_hist ]
            avg_norm = np.mean(np.linalg.norm(analogy_vecs, axis=-1))
            #avg_norm = min(0.9, np.mean(np.linalg.norm(analogy_vecs, axis=-1)))

            ##analogy_vec = np.subtract(followup_vec, nearest_vec)
            #analogy_vec = np.subtract(follow_vec, start_vec)
            #hist_analogy_vec = np.subtract(start_vec, hist_vec)
            analogy_vec = np.mean(analogy_vecs, axis=0)
            #analogy_vec = np.mean(np.vstack((analogy_vecs, hist_analogy_vecs)), axis=0)
            analogy_vec *= avg_norm / np.linalg.norm(analogy_vec)
            response_vec = np.add(sen_em, analogy_vec)

            chat_dists = sp.spatial.distance.cdist( pts, np.atleast_2d(response_vec), metric='euclidean')
            reddit_dists = sp.spatial.distance.cdist( answer_pts, np.atleast_2d(response_vec), metric='euclidean')
            inds = np.argpartition( reddit_dists.ravel(), range( len(pre_inds) ) ) if len(pre_inds != 0) else np.array([])
            chat_ind = np.argmin( chat_dists.ravel() )

            if not debug:
                response_ind = np.random.randint( len(pre_inds) )
                sys.stdout.write( answer_lines[ inds[response_ind] ] + "\n" )
            else:
                lengths = []
                for ind in range(len(pre_inds)):
                    lengths.append(sp.spatial.distance.euclidean(pts[pre_inds[ind]], pts[pre_inds[ind]+1]))
                    sys.stdout.write( "\n  chat embed " + str(ind) + ": \t" + str(dists[ pre_inds[ind] ]) + " [" + lines[ pre_inds[ind] ] + "]\n" )
                    sys.stdout.write(   "  chat follow " + str(ind) + ": \t " + str(lengths[ind]) + "  [" + lines[ pre_inds[ind] + 1 ] + "]" )
                
                sys.stdout.write( "\n\n  analogy chat: \t" + str(chat_dists[chat_ind]) + " [" + lines[ chat_ind ] + "]\n" )
                sys.stdout.write( "  average length: \t " + str(np.mean(lengths)) + "\n" )
                sys.stdout.write( "  analogy length: \t " + str(np.linalg.norm(analogy_vec)) + "\n\n" )
                
                for ind in range(len(pre_inds)):
                    sys.stdout.write( "  analogy reddit " + str(ind) + ": \t" + str(reddit_dists[inds[ind]]) + " [" + answer_lines[ inds[ind] ] + "]\n" )
                print("")
        else:
            print("\n  No viable results found.\n")

if __name__ == "__main__":
    variation = 5 if len(sys.argv) < 2 else int(sys.argv[1])

    # for i in range(10):
    #     lines, pts = embed_file( "/mnt/pccfs/backed_up/alexa_data/clean_reddit/clean{}.txt".format(i) )
    #     np.save( "/mnt/pccfs/backed_up/alexa_data/clean_reddit/universal_sentence_encodings/clean{}_pts".format(i), pts, allow_pickle=False )
    #     np.save( "/mnt/pccfs/backed_up/alexa_data/clean_reddit/universal_sentence_encodings/clean{}_lines".format(i), lines, allow_pickle=False )
    #     print("Finished file", i)

    lines = np.load( "/mnt/pccfs/not_backed_up/data/chitchat/processed/ccc_lines.npy" )
    pts = np.load( "/mnt/pccfs/not_backed_up/data/chitchat/processed/ccc_pts.npy" )

    # # THIS IS TO PARSE A REDDIT FILE INTO SHORT SENTENCES.
    # data = []
    # with open("/mnt/pccfs/backed_up/alexa_data/clean_reddit/clean0.txt", 'rb') as f:
    #     data = f.read()
    #     data = [s.strip() for s in re.split('<eos>|</s>|\n', data) if s.strip() != '']
    # with open("/mnt/pccfs/backed_up/alexa_data/clean_reddit/universal_sentence_encodings/clean0_lines.txt", 'wb') as f:
    #     lines = ""
    #     for l in data:
    #         lines += l + "\n"
    #     f.write(lines.strip())

    # lines, pts = embed_file( "/mnt/pccfs/backed_up/alexa_data/clean_reddit/universal_sentence_encodings/small0_lines.txt")
    # np.save("/mnt/pccfs/backed_up/alexa_data/clean_reddit/universal_sentence_encodings/clean0_pts", pts, allow_pickle=False )
    # np.save("/mnt/pccfs/backed_up/alexa_data/clean_reddit/universal_sentence_encodings/clean0_lines", lines, allow_pickle=False )
    
    answer_pts = np.load("/mnt/pccfs/backed_up/alexa_data/clean_reddit/universal_sentence_encodings/clean0_pts.npy")
    answer_lines = np.load("/mnt/pccfs/backed_up/alexa_data/clean_reddit/universal_sentence_encodings/clean0_lines.npy")

    simple_chat( lines, pts, answer_lines, answer_pts, variation=variation )