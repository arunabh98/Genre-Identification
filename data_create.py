import os
import sys
import sqlite3
import numpy as np
from operator import itemgetter
import hdf5_getters as GETTERS
GENRES=('classic pop and rock','punk','folk','pop','dance and electronica','metal','jazz and blues','classical','hip-hop','soul and reggae')


def path_from_trackid(msddir,trackid):
    """
    Create a full path from the main MSD dir and a track id.
    Does not check if the file actually exists.
    """
    p = os.path.join(msddir,trackid[2])
    p = os.path.join(p,trackid[3])
    p = os.path.join(p,trackid[4])
    p = os.path.join(p,trackid.upper()+'.h5')
    return p


def feat_names():
    """ return the name of each feature return by the following function """
    # basic global info
    res =  ['track_id','artist_name','title','loudness','tempo','time_signature','key','mode','duration']
    # avg timbre, var timbre
    for k in range(1,13):
        res.append( 'avg_timbre'+str(k))
    for k in range(1,13):
        res.append( 'var_timbre'+str(k))
    for k in range(1,13):
        res.append( 'max_segment_timbre'+str(k))
    # done
    return res


def feat_from_file(path):
    """
    Extract a list of features in an array, already converted to string
    """
    feats = []
    h5 = GETTERS.open_h5_file_read(path)
    # basic info
    feats.append( GETTERS.get_track_id(h5) )
    feats.append( GETTERS.get_artist_name(h5).replace(',','') )
    feats.append( GETTERS.get_title(h5).replace(',','') )
    feats.append( GETTERS.get_loudness(h5) )
    feats.append( GETTERS.get_tempo(h5) )
    feats.append( GETTERS.get_time_signature(h5) )
    feats.append( GETTERS.get_key(h5) )
    feats.append( GETTERS.get_mode(h5) )
    feats.append( GETTERS.get_duration(h5) )

    segments_loudness = np.asarray(GETTERS.get_segments_loudness_max(h5))
    max_segment_indice = np.argmax(segments_loudness)
    # timbre
    timbre = GETTERS.get_segments_timbre(h5)
    max_segment_timbre = timbre[max_segment_indice, :]
    avg_timbre = np.average(timbre,axis=0)
    for k in avg_timbre:
        feats.append(k)
    var_timbre = np.var(timbre,axis=0)
    for k in var_timbre:
        feats.append(k)
    for k in max_segment_timbre:
        feats.append(k)
    # done with h5 file
    h5.close()
    # makes sure we return strings
    feats = map(lambda x: str(x), feats)
    return feats



def die_with_usage():
    """ HELP MENU """
    print 'create_genre_dataset.py'
    print '   by T. Bertin-Mahieux (2011) Columbia University'
    print '      tb2332@columbia.edu'
    print ''
    print 'USAGE'
    print '   ./create_genre_dataset.py <msd dir> <tm.db> <artist_tag.db> <output>'
    print 'PARAMS'
    print '         msd dir   - main directory "/data" of the MSD'
    print '           tm.db   - track_metadata.db'
    print '   artist_tag.db   - SQLite db with terms and tags'
    print '          output   - output dataset in a text file'
    sys.exit(0)


if __name__ == '__main__':

    # help menu
    if len(sys.argv) < 5:
        die_with_usage()

    # params
    msddir = sys.argv[1]
    tmdb = sys.argv[2]
    atdb = sys.argv[3]
    outputf = sys.argv[4]

    # sanity checks
    if not os.path.isfile(tmdb):
        print 'ERROR:',tmdb,'does not exist.'
        sys.exit(0)
    if not os.path.isfile(atdb):
        print 'ERROR:',atdb,'does not exist.'
        sys.exit(0)
    if os.path.isfile(outputf):
        print 'ERROR:',outputf,'already exists.'
        sys.exit(0)

    # open SQLite connections
    conn_tm = sqlite3.connect(tmdb)
    conn_at = sqlite3.connect(atdb)

    # get top 50 most used musicbrainz tags
    # makes sure the ones we selected are in the top 50
    q = "SELECT mbtag,Count(artist_id) FROM artist_mbtag GROUP BY mbtag"
    res = conn_at.execute(q)
    top50mbtags = sorted(res.fetchall(),key=itemgetter(1),reverse=True)[:50]
    top50mbtags_names = map(lambda x: x[0], top50mbtags)
    for g in GENRES:
        assert g in top50mbtags_names,'Wrong or unrecognized genre: '+str(g)

    # for each of these genre, select the artists for which this is the
    # most applied genre (among the 10)
    genre_artists = {}
    for genre in GENRES:
        genre_artists[genre] = set()
        q = "SELECT artist_id FROM artist_mbtag WHERE mbtag='"+genre+"'"
        res = conn_at.execute(q)
        artists = map(lambda x: x[0], res.fetchall())
        for a in artists:
            q = "SELECT mbtag FROM artist_mbtag WHERE artist_id='"+a+"'"
            res = conn_at.execute(q)
            mbtags = map(lambda x: x[0], res.fetchall())
            artist_is_safe = True
            # for g2 in top50mbtags_names:
            #     if g2 != genre and g2 in mbtags:
            #         #print 'artist:',a,'we got both',genre,'and',g2
            #         artist_is_safe = False; break
            if artist_is_safe:
                genre_artists[genre].add( a )
    print 'number of safe artists for each genre:'
    for g in genre_artists.keys():
        print g,'->',len(genre_artists[g])

    # how many songs?
    cnt_total = 0
    for genre in GENRES:
        cnt = 0
        artists = genre_artists[genre]
        for artist in artists:
            q = "SELECT Count(track_id) FROM songs WHERE artist_id='"+artist+"'"
            res = conn_tm.execute(q)
            cnt += res.fetchone()[0]
        print 'NUMBER OF SONGS FOR:',genre,'->',cnt
        cnt_total += cnt
    print '**************************'
    print 'Total number of songs:',cnt_total

    # let's roll!
    # open output file, put intro
    output = open(outputf,'w')

    # feats name
    feats_name = feat_names()
    output.write('genre')
    for k in range(len(feats_name)):
        output.write(','+feats_name[k])
    output.write('\n')

    # iterate over all songs
    cnt_missing = 0 # debugging thing to limit printouts on my laptop
    for genre in GENRES:
        cnt = 0
        artists = genre_artists[genre]
        for artist in artists:
            q = "SELECT track_id FROM songs WHERE artist_id='"+artist+"'"
            res = conn_tm.execute(q)
            track_ids = map(lambda x: x[0], res.fetchall())
            for path in map(lambda x: path_from_trackid(msddir,x),track_ids):
                if not os.path.isfile(path):
                    cnt_missing += 1
                    if cnt_missing < 10:
                        print 'ERROR:',path,'does not exist.'
                    continue
                feats = feat_from_file(path)
                assert len(feats) == len(feats_name),'feat length problem, len(feats)='+str(len(feats))
                output.write(genre)
                for k in range(len(feats)):
                    output.write(','+feats[k])
                output.write('\n')
                

    # close output file
    output.close()

    # os.remove("output.csv")

    # close SQLite connections
    conn_tm.close()
    conn_at.close()
