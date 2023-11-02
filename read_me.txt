#A Hello, this is a preview version.

#B We have submitted the preview of our model
1. Glo_MLM.py.  # The model in the paper.
2. globle_coocurrence.py # Script for compute counts.


#C Implementation pipeline.
1. Compute n-co-occurrence matrix. See #C
2. Retrive token co-occurrence of the input sequence from n-co-occurrence matrix. See generalization_of_masking/initialization.py/def glo_fn().
     e.g., we input [x1,x2,x3,x4,x4], and the n-co-occurrence matrix is X. We need the token co-occurrence of the input sequence:
     tc = [ [X11,X12,X13,X14,X15],
         [X21,X22,X23,X24,X25],
         [X31,X32,X33,X34,X35],
         [X41,X42,X43,X44,X45],
         [X51,X52,X53,X54,X55] ]
3. Howerver, we only take into account n-neighbors or n-gram, as discussed in Eq. 3. So, in Glo_MLM.py line 112-118.
    we computer n-gram masking to mask above matrix.
    e.g., we compute a 5-gram masking:
     masking = [ [0,1,1,0,0],
                 [1,0,1,1,0],
                 [1,1,0,1,1],
                 [0,1,1,0,1],
                 [0,0,1,1,1] ]
     Note that we do not count token-self co-occurrence.
     Then, for the globle coocurrence modeling, the label is:
        gc = tc * masking = [ [0,X12,X13,0,0],
                              [X11,0,X13,X14,0],
                              [X31,X32,0,X34,X35],
                              [0,X42,X43,0,X45],
                              [0,0,X53,X54,0] ]

4. We organize the input for the globle coocurrence modeling in Glo_MLM.py line 124.
    e.g., suppose the hide state of the token is h,
          We need to predict x1 and x3. So, we have MLM_output h= [h1,0,0,0,0,   and  output_O o= [0,h2,h3,0,0,
                                                                  0,0,0,0,0,                       0,0,0,0,0,
                                                                  0,0,h3,0,0,                      h1,h2,0,h4,h5,
                                                                  0,0,0,0,0,                       0,0,0,0,0,
                                                                  0,0,0,0,0, ]                      0,0,0,0,0,]

          Then, we do in Glo_MLM.py line 133:
          a.    output_matirx = h * oT =[ [0,h1h2,h1h3,0,0],
                                                 [0,0,0,0,0],
                                                 [h3h1,h3h2,0,h3h4,h3h5],
                                                 [0,0,0,0,0],
                                                 [0,0,0,0,0]]
         b. we minimize between output_matirx and gc considering non-zero values in output_matirx.




#