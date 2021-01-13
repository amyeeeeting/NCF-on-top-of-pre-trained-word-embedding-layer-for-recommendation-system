# NCF on top of-pre-trained word embedding layer for recommendation system
This is a implementation of Neural Collaborative Filtering with pre-trained word embedding. Using user_id and its watched item_id (before the time split) and each item_id description as input to train the model. 
Ranking item_id for each user_id and compare the result with those item_id that have watched by the user_id (after the time split) to evaluate the ranking result.
To evaluate the result, I use mean reciprocal rank(mrr) as the final output.

# Input & Output
Input: 
a. A dictionary of user id : [list of item_id]

b. a dictionary of item_id： item's descriptions(str)
       
Output: mrr of the NCF model

# result
mrr without word embedding: 0.03915

mrr with word embedding : 0.04581
